"""
perturbation_manager.py
------------------------
A modular perturbation framework for visual input streams in autonomous driving agents.
Designed for use with the CARLA simulator and easily extensible for new perturbation types
or input formats (e.g. LiDAR, depth maps, adversarial patches).

Usage example:
    pm = PerturbationManager()
    perturbed = pm.perturb_wide_image(wide_rgbs, perturbation="gaussian_noise", intensity=5)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Callable, Literal

from ATOMs_Analysis.atoms_config import ExperimentConfig


# ---------------------------------------------------------------------------
# Registry decorator — used to register perturbation functions automatically
# ---------------------------------------------------------------------------

_WIDE_IMAGE_REGISTRY: dict[str, Callable] = {}


def _register_wide(name: str):
    """Decorator that registers a function under a given perturbation name."""
    def decorator(fn: Callable) -> Callable:
        _WIDE_IMAGE_REGISTRY[name] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# PerturbationManager
# ---------------------------------------------------------------------------

class PerturbationManager:
    """
    Central manager for applying perturbations to agent visual input streams.

    Each perturbation is registered via the @_register_wide decorator and can
    be invoked by name through the public perturb_wide_image() method.

    Attributes
    ----------
    verbose : bool
        If True, prints which perturbation is being applied on each call.

    Extending
    ---------
    To add a new wide-image perturbation, define a module-level function with
    the signature:

        @_register_wide("my_perturbation_name")
        def _my_perturbation(
            wide_rgbs: List[np.ndarray],
            intensity: float,
            **kwargs
        ) -> List[np.ndarray]:
            ...
            return wide_rgbs  # always return a list of three arrays

    Then call it via:
        pm.perturb_wide_image(wide_rgbs, perturbation="my_perturbation_name", intensity=...)

    For other input modalities (LiDAR, depth, etc.), add a new registry dict
    and a corresponding public method following the same pattern.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # For fgsm speedup
        self.frame_counter = 0
        self.attack_interval = ExperimentConfig.FRAMES_TO_SKIP + 1  # Recompute attack every 3rd frame
        self.last_wide_noise = None
        self.last_narr_noise = None
        self.last_tfv6_noise = None   # cached δ for pgd_attack_tfv6

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def perturb_wide_image(
        self,
        wide_rgbs: List[np.ndarray],
        perturbation: str,
        intensity: float,
        camera_index: int | None = None,
        **kwargs,
    ) -> List[np.ndarray]:
        """
        Apply a named perturbation to the list of wide RGB camera images.

        Parameters
        ----------
        wide_rgbs : List[np.ndarray]
            List of three HxWxC uint8 arrays, one per camera.
        perturbation : str
            Name of the perturbation to apply (see list_perturbations()).
        intensity : float
            Scalar controlling the strength of the perturbation.
            Semantics depend on the chosen perturbation type (see each
            function's docstring below).
        camera_index : int | None
            If provided, the perturbation is applied only to that camera
            (0, 1, or 2). If None, it is applied to all cameras.
        **kwargs
            Additional keyword arguments forwarded to the perturbation function.

        Returns
        -------
        List[np.ndarray]
            Perturbed list of three HxWxC uint8 arrays.

        Raises
        ------
        ValueError
            If `perturbation` is not a registered perturbation name.
        """
        if perturbation not in _WIDE_IMAGE_REGISTRY:
            available = ", ".join(self.list_perturbations())
            raise ValueError(
                f"Unknown perturbation '{perturbation}'. "
                f"Available: {available}"
            )

        if self.verbose:
            cam_info = f"camera {camera_index}" if camera_index is not None else "all cameras"
            print(f"[PerturbationManager] Applying '{perturbation}' "
                  f"(intensity={intensity}) to {cam_info}.")

        fn = _WIDE_IMAGE_REGISTRY[perturbation]

        # Apply per-camera or globally
        if camera_index is not None:
            target = [wide_rgbs[camera_index]]
            result = fn(target, intensity, **kwargs)
            output = list(wide_rgbs)       # shallow copy of the list
            output[camera_index] = result[0]
            return output
        else:
            return fn(list(wide_rgbs), intensity, **kwargs)

    def perturb_narrow_image(
        self,
        narr_rgb: np.ndarray,
        perturbation: str,
        intensity: float,
        **kwargs,
    ) -> np.ndarray:
        """
        Apply a named perturbation to the single narrow camera image.

        Reuses the same registry as perturb_wide_image by wrapping the
        image in a temporary list. Perturbations that are multi-camera
        specific (camera_swap, camera_loss) will no-op or warn gracefully.

        Parameters
        ----------
        narr_rgb : np.ndarray
            Raw narrow camera image, shape [H, W, C] (e.g. [240, 384, 4]).
            The 4th alpha channel is ignored by all perturbations.
        perturbation : str
            Name of the perturbation to apply.
        intensity : float
            Strength parameter — same semantics as perturb_wide_image.

        Returns
        -------
        np.ndarray
            Perturbed image with the same shape as the input.
        """
        if perturbation not in _WIDE_IMAGE_REGISTRY:
            available = ", ".join(self.list_perturbations())
            raise ValueError(
                f"Unknown perturbation '{perturbation}'. "
                f"Available: {available}"
            )

        if self.verbose:
            print(f"[PerturbationManager] Applying '{perturbation}' "
                  f"(intensity={intensity}) to narrow camera.")

        fn = _WIDE_IMAGE_REGISTRY[perturbation]
        result = fn([narr_rgb], intensity, **kwargs)
        return result[0]
    
    def pgd_attack(
        self,
        model,
        wide_rgbs_: "torch.Tensor",
        narr_rgb_: "torch.Tensor",
        cmd_value: "torch.Tensor",
        target: Literal["steer_right", "steer_left", "brake", "max_steer"] = "steer_right",
        epsilon: float = 8.0,
        n_steps: int = 15,
        step_size: float = None,
        random_start: bool = True,
        apply_to_wide: bool = True,
        apply_to_narrow: bool = True,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Projected Gradient Descent (PGD) adversarial attack.

        Iteratively refines an adversarial perturbation δ by taking n_steps
        gradient ascent steps of size α, projecting back into the ℓ∞ ball of
        radius ε after each step:

            δ₀  = U(−ε, ε)   if random_start else 0
            δₜ₊₁ = Π_ε [ δₜ + α · sign(∇_δ L(x + δₜ)) ]
            x_adv = clip( x + δ_n, 0, 255 )

        Within the same ε budget PGD is substantially stronger than FGSM
        because each small step stays close to the loss surface, accumulating
        curvature information that a single coarse step discards.

        Parameters
        ----------
        model : object
            Policy model with a callable `.policy(wide, narr, cmd)` returning
            (steer_logits, throt_logits, brake_logits).
        wide_rgbs_ : torch.Tensor
            Wide camera tensor, shape [1, 3, H, W], float32, range [0, 255].
        narr_rgb_ : torch.Tensor
            Narrow camera tensor, shape [1, 3, H, W], float32, range [0, 255].
        cmd_value : torch.Tensor
            Command value passed to the policy.
        target : str
            Attack objective — same options as fgsm_attack:
            "steer_right", "steer_left", "brake", "max_steer".
        epsilon : float
            ℓ∞ perturbation budget in pixel units [0, 255].
            At epsilon=4 the perturbation is virtually invisible;
            epsilon=8 matches a common imperceptibility threshold in the
            adversarial ML literature.
        n_steps : int
            Number of PGD iterations. More steps → stronger attack at the
            cost of n_steps forward+backward passes per frame.
            Typical values: 7 (fast), 20 (strong), 40 (near-optimal within ε).
        step_size : float or None
            Gradient step size α per iteration in pixel units.
            Defaults to 2.5 * epsilon / n_steps, which is the standard
            heuristic from Madry et al. and tends to work well in practice.
            Override only if you observe the optimisation oscillating.
        random_start : bool
            If True, initialise δ from U(−ε, ε) rather than from 0.
            This is strongly recommended: a zero start can land in a locally
            flat region and waste the first several steps.
        apply_to_wide : bool
            Whether to perturb the wide image tensor.
        apply_to_narrow : bool
            Whether to perturb the narrow image tensor.

        Returns
        -------
        (wide_adv, narr_adv) : tuple[torch.Tensor, torch.Tensor]
            Adversarially perturbed tensors, same shape and device as inputs,
            pixel values clipped to [0, 255].

        Notes
        -----
        Drop-in replacement for fgsm_attack — call it the same way:

            wide_adv, narr_adv = pm.pgd_attack(
                self.image_model, wide_rgbs_, narr_rgb_,
                cmd_value, target="brake", epsilon=8.0, n_steps=20
            )
            steer_logits, throt_logits, brake_logits = \\
                self.image_model.policy(wide_adv, narr_adv, cmd_value)
        """
        if step_size is None:
            step_size = 2.5 * epsilon / n_steps

        if self.frame_counter % self.attack_interval == 0:
            with torch.enable_grad():

                # ---- 1. Initialise perturbation tensors --------------------------
                if apply_to_wide:
                    if random_start:
                        delta_wide = torch.empty_like(wide_rgbs_).uniform_(-epsilon, epsilon)
                    else:
                        delta_wide = torch.zeros_like(wide_rgbs_)
                    delta_wide.requires_grad_(True)

                if apply_to_narrow:
                    if random_start:
                        delta_narr = torch.empty_like(narr_rgb_).uniform_(-epsilon, epsilon)
                    else:
                        delta_narr = torch.zeros_like(narr_rgb_)
                    delta_narr.requires_grad_(True)

                was_training = model.training
                model.eval()

                # ---- 2. PGD iterations -------------------------------------------
                for step in range(n_steps):

                    # Build perturbed inputs for this step
                    wide_in = (wide_rgbs_ + delta_wide) if apply_to_wide else wide_rgbs_
                    narr_in = (narr_rgb_ + delta_narr) if apply_to_narrow else narr_rgb_

                    steer_logits, throt_logits, brake_logits = model.policy(
                        wide_in, narr_in, cmd_value
                    )

                    # Targeted loss — identical semantics to fgsm_attack
                    if target == "steer_right":
                        loss = -steer_logits
                    elif target == "steer_left":
                        loss = steer_logits
                    elif target == "brake":
                        loss = -brake_logits
                    elif target == "max_steer":
                        loss = -steer_logits.abs()
                    else:
                        raise ValueError(
                            f"Unknown PGD target '{target}'. "
                            "Choose from: steer_right, steer_left, brake, max_steer."
                        )

                    loss.sum().backward()

                    with torch.no_grad():
                        # Gradient ascent step + project back into ε-ball
                        if apply_to_wide:
                            delta_wide.data = torch.clamp(
                                delta_wide + step_size * delta_wide.grad.sign(),
                                -epsilon, epsilon,
                            )
                            delta_wide.grad.zero_()

                        if apply_to_narrow:
                            delta_narr.data = torch.clamp(
                                delta_narr + step_size * delta_narr.grad.sign(),
                                -epsilon, epsilon,
                            )
                            delta_narr.grad.zero_()

                    if self.verbose:
                        print(f"[PerturbationManager] PGD | target={target} | "
                              f"step={step + 1}/{n_steps} | ε={epsilon} | "
                              f"loss={loss.item():.4f}")

                model.train(was_training)

                # Cache the final noise masks for use across attack_interval frames
                if apply_to_wide:
                    self.last_wide_noise = delta_wide.detach()
                if apply_to_narrow:
                    self.last_narr_noise = delta_narr.detach()

        self.frame_counter += 1

        # ---- 3. Apply cached perturbation to clean inputs ----------------
        wide_adv = wide_rgbs_.detach().clone()
        narr_adv = narr_rgb_.detach().clone()

        if apply_to_wide and self.last_wide_noise is not None:
            wide_adv = torch.clamp(wide_adv + self.last_wide_noise, 0.0, 255.0)

        if apply_to_narrow and self.last_narr_noise is not None:
            narr_adv = torch.clamp(narr_adv + self.last_narr_noise, 0.0, 255.0)

        return wide_adv, narr_adv

    def pgd_attack_tfv6(
        self,
        nets: list,
        data: dict,
        target: Literal["brake", "max_speed", "steer_left", "steer_right"] = "brake",
        epsilon: float = 8.0,
        n_steps: int = 10,
        step_size: float = None,
        random_start: bool = True,
    ) -> "torch.Tensor":
        """
        PGD adversarial attack for the TFV6 agent.

        Backpropagates through `net.forward(data)` — specifically through
        `pred_target_speed_distribution` (raw 8-bin speed logits) or
        `pred_future_waypoints` — to craft a perturbation δ on the RGB tensor.

        The gradient flows only through `data["rgb"]`; all other inputs
        (LiDAR, target points, command, speed) are held fixed.

        Parameters
        ----------
        nets : list[TFv6]
            The TFV6 model ensemble (``self.closed_loop_inference.nets``).
            Gradients are averaged across all ensemble members.
        data : dict
            Tensor dict as built in ``SensorAgent.run_step`` — must contain
            key ``"rgb"`` with shape ``[1, 3, H, W]``, dtype float32.
        target : str
            Attack objective:
            - ``"brake"``       — maximise P(speed bin 0 = 0 m/s) → force stop.
            - ``"max_speed"``   — maximise P(speed bin 7 = 20 m/s) → force max speed.
            - ``"steer_left"``  — minimise mean predicted waypoint x (push left).
            - ``"steer_right"`` — maximise mean predicted waypoint x (push right).
        epsilon : float
            ℓ∞ pixel budget (same scale as ``data["rgb"]``, i.e. 0–255).
        n_steps : int
            PGD iterations.
        step_size : float | None
            Per-step size; defaults to 2.5 * epsilon / n_steps (Madry heuristic).
        random_start : bool
            Initialise δ from U(−ε, ε) rather than zero (strongly recommended).

        Returns
        -------
        torch.Tensor
            Adversarially perturbed RGB tensor, same shape as ``data["rgb"]``,
            pixel values clipped to [0, 255].
        """
        if step_size is None:
            step_size = 2.5 * epsilon / n_steps

        rgb_clean = data["rgb"]   # [1, 3, H, W] float32, no grad

        if self.frame_counter % self.attack_interval == 0:
            with torch.enable_grad():
                # Initialise perturbation δ
                if random_start:
                    delta = torch.empty_like(rgb_clean).uniform_(-epsilon, epsilon)
                else:
                    delta = torch.zeros_like(rgb_clean)
                delta = delta.detach().requires_grad_(True)

                was_training = [net.training for net in nets]
                for net in nets:
                    net.eval()

                for step in range(n_steps):
                    data_adv = {**data, "rgb": torch.clamp(rgb_clean.detach() + delta, 0.0, 255.0)}

                    total_loss = torch.tensor(0.0, device=rgb_clean.device)
                    for net in nets:
                        pred = net.forward(data_adv)

                        if target in ("brake", "max_speed"):
                            logits = pred.pred_target_speed_distribution   # [B, 8]
                            tgt_bin = 0 if target == "brake" else 7
                            tgt_tensor = torch.full(
                                (logits.shape[0],), tgt_bin,
                                dtype=torch.long, device=logits.device,
                            )
                            loss = torch.nn.functional.cross_entropy(logits, tgt_tensor)
                        elif target == "steer_left":
                            # Minimise mean waypoint x → steer left
                            loss = pred.pred_future_waypoints[..., 0].mean()
                        elif target == "steer_right":
                            # Maximise mean waypoint x → steer right
                            loss = -pred.pred_future_waypoints[..., 0].mean()
                        else:
                            raise ValueError(
                                f"Unknown PGD TFV6 target '{target}'. "
                                "Choose: brake, max_speed, steer_left, steer_right."
                            )
                        total_loss = total_loss + loss

                    (total_loss / len(nets)).backward()

                    with torch.no_grad():
                        delta.data = torch.clamp(
                            delta + step_size * delta.grad.sign(),
                            -epsilon, epsilon,
                        )
                        delta.grad.zero_()

                    if self.verbose:
                        print(
                            f"[PerturbationManager] PGD-TFV6 | target={target} | "
                            f"step={step + 1}/{n_steps} | ε={epsilon} | "
                            f"loss={total_loss.item():.4f}"
                        )

                for net, training in zip(nets, was_training):
                    net.train(training)

                self.last_tfv6_noise = delta.detach()

        self.frame_counter += 1

        if self.last_tfv6_noise is None:
            return rgb_clean.detach().clone()

        return torch.clamp(rgb_clean.detach() + self.last_tfv6_noise, 0.0, 255.0)

    def fgsm_attack(
        self,
        model,
        wide_rgbs_: "torch.Tensor",
        narr_rgb_: "torch.Tensor",
        cmd_value: "torch.Tensor",
        target: Literal["steer_right", "steer_left", "brake", "max_steer"] = "steer_right",
        epsilon: float = 8.0,
        apply_to_wide: bool = True,
        apply_to_narrow: bool = True,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Fast Gradient Sign Method (FGSM) adversarial attack.

        Computes one forward + one backward pass through the policy network
        to obtain ∇ₓ L, then applies:

            x_adv = clip( x + ε · sign(∇ₓ L), 0, 255 )

        The loss L is defined by `target` — see below.

        Parameters
        ----------
        model : object
            The image model with a callable `.policy(wide, narr, cmd)` method
            that returns (steer_logits, throt_logits, brake_logits).
        wide_rgbs_ : torch.Tensor
            Wide camera tensor, shape [1, 3, H, W], float32, range [0, 255].
            As produced by the existing pipeline before any model call.
        narr_rgb_ : torch.Tensor
            Narrow camera tensor, shape [1, 3, H, W], float32, range [0, 255].
        cmd_value : torch.Tensor
            Command value as passed to the original policy call.
        target : str
            Defines what the attack optimises for:
            - "steer_right" : maximise steer_logits → hard right turn.
            - "steer_left"  : minimise steer_logits → hard left turn.
            - "brake"       : maximise brake_logits → force emergency stop.
            - "max_steer"   : maximise |steer_logits| in whichever direction
                              the gradient already favours — fastest to diverge.
        epsilon : float
            Perturbation budget in raw pixel units (same scale as the input,
            i.e. [0, 255]).  A value of 8.0 is a good starting point; increase
            to 16 or 32 if the agent is not yet destabilised.
        apply_to_wide : bool
            Whether to perturb the wide image tensor. Default True.
        apply_to_narrow : bool
            Whether to perturb the narrow image tensor. Default True.

        Returns
        -------
        (wide_adv, narr_adv) : tuple[torch.Tensor, torch.Tensor]
            Adversarially perturbed tensors with the same shape and device as
            the inputs. Pixel values are clipped to [0, 255].

        Notes
        -----
        Call this method *instead of* the normal model call, then pass the
        returned tensors to the real policy call:

            wide_adv, narr_adv = pm.fgsm_attack(
                self.image_model, wide_rgbs_, narr_rgb_,
                cmd_value, target="steer_right", epsilon=8.0
            )
            steer_logits, throt_logits, brake_logits = \\
                self.image_model.policy(wide_adv, narr_adv, cmd_value)
        """
        if self.frame_counter % self.attack_interval == 0:
            with torch.enable_grad():

                # ---- 1. Prepare differentiable copies --------------------------------
                wide_in = wide_rgbs_.detach().clone()
                narr_in = narr_rgb_.detach().clone()

                if apply_to_wide:
                    wide_in.requires_grad_(True)
                if apply_to_narrow:
                    narr_in.requires_grad_(True)

                # ---- 2. Forward pass -------------------------------------------------
                # Temporarily set model to eval and disable dropout / BN noise
                was_training = model.training
                model.eval()
                steer_logits, throt_logits, brake_logits = model.policy(
                    wide_in, narr_in, cmd_value
                )
                model.train(was_training)


                # ---- 3. Targeted loss ------------------------------------------------
                if target == "steer_right":
                    # Maximise steer → push output as far right as possible
                    loss = -steer_logits
                elif target == "steer_left":
                    # Minimise steer → push output as far left as possible
                    loss = steer_logits
                elif target == "brake":
                    # Maximise brake probability → force an emergency stop
                    loss = -brake_logits
                elif target == "max_steer":
                    # Maximise absolute steer in whichever direction gradient favours
                    loss = -steer_logits.abs()
                else:
                    raise ValueError(
                        f"Unknown FGSM target '{target}'. "
                        "Choose from: steer_right, steer_left, brake, max_steer."
                    )

                if self.verbose:
                    print(f"[PerturbationManager] FGSM | target={target} | "
                          f"ε={epsilon} | loss={loss.item():.4f}")

                # ---- 4. Backward pass ------------------------------------------------
                loss.sum().backward()

                # ---- Store Adversarial Mask ------------------------------------------
                if apply_to_wide:
                    self.last_wide_noise = wide_in.grad.sign()
                if apply_to_narrow:
                    self.last_narr_noise = narr_in.grad.sign()

        self.frame_counter += 1

        # ---- 5. Apply perturbation -------------------------------------------
        wide_adv = wide_rgbs_.detach().clone()
        narr_adv = narr_rgb_.detach().clone()

        if apply_to_wide and self.last_wide_noise is not None:
            wide_adv = torch.clamp(
                wide_adv + epsilon * self.last_wide_noise, 0.0, 255.0
            )

        if apply_to_narrow and self.last_narr_noise is not None:
            narr_adv = torch.clamp(
                narr_adv + epsilon * self.last_narr_noise, 0.0, 255.0
            )

        return wide_adv, narr_adv

    def perturb_tfv6_image(
        self,
        rgb_chw: np.ndarray,
        perturbation: str,
        intensity: float,
        camera_index: int | None = None,
        n_cameras: int = 6,
    ) -> np.ndarray:
        """
        Apply a perturbation to a TFV6 concatenated RGB image [3, H, W_total].

        Splits the concatenated frame into n_cameras per-camera sub-images,
        applies the same registry function as perturb_wide_image (which expects
        a list of [H, W_per_cam, 3] arrays), then re-concatenates.

        This fixes the layout issues that arise when passing a single-element
        list to perturb_wide_image:
        - camera_loss: int(intensity) now indexes correctly into 0..n_cameras-1
        - camera_swap: swaps cameras 0 and n_cameras-1 (first and last)
        - mirror_horizontal: flips each camera individually, not the whole strip
        - phantom_obstacle: box centred within each camera, not across the strip

        Parameters
        ----------
        rgb_chw : np.ndarray
            TFV6 concatenated image, shape [3, H, W_total], uint8.
        perturbation : str
            Name of the perturbation (same names as perturb_wide_image).
        intensity : float
            Perturbation strength (same semantics as perturb_wide_image).
        camera_index : int | None
            If provided, perturbation is applied only to that camera
            (0..n_cameras-1). If None, applied to all cameras.
        n_cameras : int
            Number of cameras concatenated horizontally (default: 6 for TFV6).

        Returns
        -------
        np.ndarray
            Perturbed image, shape [3, H, W_total], uint8.
        """
        if perturbation not in _WIDE_IMAGE_REGISTRY:
            available = ", ".join(self.list_perturbations())
            raise ValueError(
                f"Unknown perturbation '{perturbation}'. "
                f"Available: {available}"
            )

        if self.verbose:
            cam_info = f"camera {camera_index}" if camera_index is not None else "all cameras"
            print(f"[PerturbationManager] TFV6 — applying '{perturbation}' "
                  f"(intensity={intensity}) to {cam_info}.")

        # [3, H, W_total] → [H, W_total, 3] then split per camera
        rgb_hwc = np.ascontiguousarray(rgb_chw.transpose(1, 2, 0))
        h, w_total, _ = rgb_hwc.shape
        w_per_cam = w_total // n_cameras
        cameras = [
            rgb_hwc[:, i * w_per_cam:(i + 1) * w_per_cam, :]
            for i in range(n_cameras)
        ]

        fn = _WIDE_IMAGE_REGISTRY[perturbation]

        if camera_index is not None:
            target = [cameras[camera_index]]
            cameras[camera_index] = fn(target, intensity)[0]
        else:
            cameras = fn(cameras, intensity)

        # Re-concatenate → [H, W_total, 3] → [3, H, W_total]
        rgb_hwc_out = np.concatenate(cameras, axis=1)
        return np.ascontiguousarray(rgb_hwc_out.transpose(2, 0, 1)).astype(np.uint8)

    @staticmethod
    def list_perturbations() -> List[str]:
        """Return the names of all registered wide-image perturbations."""
        return sorted(_WIDE_IMAGE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in wide-image perturbations
# ---------------------------------------------------------------------------

@_register_wide("gaussian_noise")
def _gaussian_noise(
    wide_rgbs: List[np.ndarray],
    intensity: float,
    **kwargs,
) -> List[np.ndarray]:
    """
    Add zero-mean Gaussian noise to each camera image.

    Parameters
    ----------
    intensity : float
        Standard deviation (σ) of the Gaussian noise in pixel units.
        Typical range: 1 (barely visible) – 50 (heavily degraded).
        The noisy values are clipped to [0, 255].
    """
    result = []
    for img in wide_rgbs:
        noise = np.random.normal(loc=0.0, scale=intensity, size=img.shape)
        if isinstance(img, torch.Tensor):
            noisy = np.clip(img.float() + noise, 0, 255).int()
        else:
            noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        result.append(noisy)
    return result


@_register_wide("brightness_scale")
def _brightness_scale(
    wide_rgbs: List[np.ndarray],
    intensity: float,
    **kwargs,
) -> List[np.ndarray]:
    """
    Multiply every pixel value by a constant factor.

    Parameters
    ----------
    intensity : float
        Multiplicative factor applied to all pixel values.
        Values < 1.0 darken the image; values > 1.0 brighten it.
        Results are clipped to [0, 255].
        Example: 0.5 (half brightness), 1.5 (50 % brighter).
    """
    result = []
    for img in wide_rgbs:
        if isinstance(img, torch.Tensor):
            scaled = np.clip(img.float() * intensity, 0, 255).int()
        else:
            scaled = np.clip(img.astype(np.float32) * intensity, 0, 255).astype(np.uint8)
        result.append(scaled)
    return result


@_register_wide("camera_loss")
def _camera_loss(
    wide_rgbs: List[np.ndarray],
    intensity: float,
    **kwargs,
) -> List[np.ndarray]:
    """
    Simulate the complete loss of one camera by replacing its image with zeros.

    The camera to black out is determined by `intensity`, interpreted as a
    camera index (0, 1, or 2). If intensity is not a valid index the
    perturbation is a no-op and a warning is printed.

    Parameters
    ----------
    intensity : float
        Camera index to drop (0, 1, or 2). Non-integer values are truncated.
        When used via perturb_wide_image(..., camera_index=k), prefer
        passing `intensity=0` and letting `camera_index` select the target —
        or use intensity directly to name the lost camera.

    Notes
    -----
    Typical invocation:
        pm.perturb_wide_image(wide_rgbs, perturbation="camera_loss",
                              intensity=1)   # drops camera 1
    or equivalently:
        pm.perturb_wide_image(wide_rgbs, perturbation="camera_loss",
                              intensity=0, camera_index=1)
    """
    idx = int(intensity)
    if idx not in range(len(wide_rgbs)):
        print(f"[PerturbationManager] Warning: camera_loss index {idx} is out "
              f"of range for a list of {len(wide_rgbs)} cameras. No-op.")
        return wide_rgbs

    result = list(wide_rgbs)
    result[idx] = np.zeros_like(wide_rgbs[idx])
    return result


@_register_wide("isolate_channel")
def _isolate_channel(
    wide_rgbs: List[np.ndarray],
    intensity: float,
    **kwargs,
) -> List[np.ndarray]:
    """
    Isolate a single colour channel by zeroing out all others.

    Useful for debugging colour-mapping issues (e.g. BGR vs RGB confusion):
    pass each channel index in turn and observe which one lights up for a
    known-colour object to determine the actual channel order of your images.

    Parameters
    ----------
    intensity : float
        Index of the channel to keep (0, 1, or 2). Non-integer values are
        truncated.  With a standard 3-channel image:
            0 → first channel   (R if RGB, B if BGR)
            1 → second channel  (G in both RGB and BGR)
            2 → third channel   (B if RGB, R if BGR)

    Notes
    -----
    Typical invocation — inspect each channel in turn:
        pm.perturb_wide_image(wide_rgbs, perturbation="isolate_channel", intensity=0)
        pm.perturb_wide_image(wide_rgbs, perturbation="isolate_channel", intensity=1)
        pm.perturb_wide_image(wide_rgbs, perturbation="isolate_channel", intensity=2)

    If a yellow/red object (which should have a strong R component) appears
    bright only when intensity=2, your pipeline is delivering BGR-ordered
    images, because channel 2 in BGR is R.
    """
    keep = int(intensity)
    n_channels = wide_rgbs[0].shape[2]

    if keep not in range(n_channels):
        print(f"[PerturbationManager] Warning: isolate_channel index {keep} is "
              f"out of range for images with {n_channels} channels. No-op.")
        return wide_rgbs

    result = []
    for img in wide_rgbs:
        isolated = np.zeros_like(img)
        isolated[..., keep] = img[..., keep]
        result.append(isolated)
    return result


@_register_wide("mirror_horizontal")
def _mirror_horizontal(
    wide_rgbs: List[np.ndarray],
    intensity: float,
    **kwargs,
) -> List[np.ndarray]:
    """
    Horizontally flip one or all camera images.

    This is a strong spatial perturbation: lane markings, road edges and
    any asymmetric scene structure appear mirrored, which is likely to
    confuse a steering controller trained on normal forward-facing views.
    Applied to only the leftmost or rightmost camera it creates a strong
    left/right asymmetry that may induce an unintended turn.

    `intensity` is unused (the flip is binary) but kept for API consistency.
    Pass any value, e.g. intensity=1.

    Notes
    -----
    To flip only the right camera and leave the others intact:
        pm.perturb_wide_image(wide_rgbs, perturbation="mirror_horizontal",
                              intensity=1, camera_index=2)
    """
    return [np.ascontiguousarray(np.fliplr(img)) for img in wide_rgbs]


@_register_wide("camera_swap")
def _camera_swap(
    wide_rgbs: List[np.ndarray],
    intensity: float,
    **kwargs,
) -> List[np.ndarray]:
    """
    Swap the left (index 0) and right (index 2) camera images.

    The centre camera is left untouched. From the agent's perspective the
    left half of its visual field now shows what is to its right and vice
    versa — a strong spatial contradiction that is likely to destabilise
    directional decisions.

    `intensity` and `camera_index` are both ignored for this perturbation:
    the swap is always between cameras 0 and 2 and cannot be meaningfully
    restricted to a single camera. Pass intensity=1.

    Notes
    -----
    Combining this with mirror_horizontal on the centre camera maximises
    spatial confusion while keeping every sub-image internally consistent:
        wide_rgbs = pm.perturb_wide_image(wide_rgbs, "camera_swap",    intensity=1)
        wide_rgbs = pm.perturb_wide_image(wide_rgbs, "mirror_horizontal",
                                           intensity=1, camera_index=1)
    """
    if len(wide_rgbs) < 3:
        print(f"[PerturbationManager] Warning: camera_swap requires at least 3 "
              f"cameras, got {len(wide_rgbs)}. No-op.")
        return wide_rgbs

    result = list(wide_rgbs)
    result[0], result[-1] = wide_rgbs[-1], wide_rgbs[0]
    return result


@_register_wide("blur")
def _blur(
    wide_rgbs: List[np.ndarray],
    intensity: float,
    **kwargs,
) -> List[np.ndarray]:
    """
    Apply a uniform (box) blur by averaging each pixel over its neighbours.

    Parameters
    ----------
    intensity : float
        Controls the kernel size as (2k+1) × (2k+1), where k = max(1, int(intensity)).
        intensity=1 → 3×3 kernel (mild softening)
        intensity=5 → 11×11 kernel (strong blur)
        intensity=10 → 21×21 kernel (very heavy blur)
    """
    import cv2
    k = max(1, int(intensity))
    kernel_size = 2 * k + 1  # always odd
    result = []
    for img in wide_rgbs:
        if isinstance(img, torch.Tensor):
            arr = img.numpy().astype(np.uint8)
            blurred = cv2.blur(arr, (kernel_size, kernel_size))
            result.append(torch.from_numpy(blurred))
        else:
            result.append(cv2.blur(img, (kernel_size, kernel_size)))
    return result


@_register_wide("salt_and_pepper")
def _salt_and_pepper(
    wide_rgbs: List[np.ndarray],
    intensity: float,
    **kwargs,
) -> List[np.ndarray]:
    """
    Randomly overwrite pixels with pure white (255) or pure black (0).

    Parameters
    ----------
    intensity : float
        Fraction of pixels to corrupt, in the range [0.0, 1.0].
        Half of the corrupted pixels become white (salt), half become black
        (pepper).
        intensity=0.01 →  1 % of pixels affected (light noise)
        intensity=0.05 →  5 % of pixels affected (moderate noise)
        intensity=0.20 → 20 % of pixels affected (heavy degradation)
    """
    density = float(np.clip(intensity, 0.0, 1.0))
    result = []
    for img in wide_rgbs:
        if isinstance(img, torch.Tensor):
            arr = img.numpy().astype(np.uint8)
        else:
            arr = img.copy()

        h, w = arr.shape[:2]
        n_pixels = h * w
        n_corrupt = int(n_pixels * density)

        # Salt (white)
        salt_coords = (
            np.random.randint(0, h, n_corrupt // 2),
            np.random.randint(0, w, n_corrupt // 2),
        )
        arr[salt_coords] = 255

        # Pepper (black)
        pepper_coords = (
            np.random.randint(0, h, n_corrupt // 2),
            np.random.randint(0, w, n_corrupt // 2),
        )
        arr[pepper_coords] = 0

        if isinstance(img, torch.Tensor):
            result.append(torch.from_numpy(arr))
        else:
            result.append(arr)
    return result


@_register_wide("phantom_obstacle")
def _phantom_obstacle(
    wide_rgbs: List[np.ndarray],
    intensity: float,
    **kwargs,
) -> List[np.ndarray]:
    """
    Inject a synthetic dark rectangle into the lower-centre of each camera
    image, creating strong artificial edges that mimic a nearby obstacle.

    Because the agent relies primarily on edge detection rather than colour,
    the sharp boundary of the box — not its fill value — is the adversarial
    signal. The rectangle is placed in the bottom-centre of the frame, where
    a close obstacle would project in a forward-facing camera.

    Parameters
    ----------
    intensity : float
        Controls the size of the injected rectangle as a fraction of the
        image dimensions, in the range (0.0, 1.0].
        intensity=0.1  →  box covers ~10 % of image width/height (subtle)
        intensity=0.3  →  box covers ~30 % (moderate, likely to trigger braking)
        intensity=0.6  →  box covers ~60 % (severe, dominates the lower frame)

    Notes
    -----
    The rectangle is always:
        - horizontally centred
        - anchored to the bottom edge of the image (y goes to frame bottom)
        - filled with near-black (pixel value 10) to mimic an occluding object
    Only the centre camera (index 1) is perturbed by default so the left/right
    cameras retain a consistent view, making the injected obstacle appear
    closer than it is — a more plausible and therefore more dangerous scenario.
    To perturb all cameras pass camera_index=None (or omit it).
    """
    scale = float(np.clip(intensity, 0.0, 1.0))
    result = []
    offset = 5
    for img in wide_rgbs:
        if isinstance(img, torch.Tensor):
            arr = img.numpy().astype(np.uint8)
        else:
            arr = img.copy()

        h, w = arr.shape[:2]
        box_w = int(w * scale)
        box_h = int(h * scale)

        x0 = (w - box_w) // 2
        x1 = x0 + box_w
        y0 = h - box_h - offset   # anchored to the bottom
        y1 = h - offset

        # Near-black fill — occludes the scene the way a solid obstacle would.
        # Avoid pure 0 so it doesn't alias with a camera_loss black-out.
        arr[y0:y1, x0:x1] = 20

        if isinstance(img, torch.Tensor):
            result.append(torch.from_numpy(arr))
        else:
            result.append(arr)
    return result