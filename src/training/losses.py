import torch

def compute_rotation_loss(predicted, target):
    # Compute the rotation loss between predicted and target

    #TODO: Find a better loss with SO(3) constraint
    rotation_loss = torch.nn.functional.mse_loss(predicted, target)
    return rotation_loss

def compute_translation_loss(predicted, target):
    # Compute the translation loss between predicted and target
    translation_loss = torch.nn.functional.mse_loss(predicted, target)
    return translation_loss

def compute_flow_loss(predicted, target, mask):
    # Compute the flow loss between predicted and target
    target = target.squeeze()
    flow_loss = torch.nn.functional.mse_loss(predicted, target)

    return flow_loss

def compute_loss(
    predicted_rotation, 
    target_rotation, 
    predicted_translation, 
    target_translation, 
    alpha, 
    supervise_flow=False,
    predicted_flow=None,
    gt_flow=None,
    mask=None,
    flow_alpha=0.1
):
    # Compute the weighted loss
    rotation_loss = compute_rotation_loss(predicted_rotation, target_rotation)
    translation_loss = compute_translation_loss(predicted_translation, target_translation)

    loss = {"rotation_loss": rotation_loss, "translation_loss": translation_loss}

    total_loss = alpha * rotation_loss + (alpha) * translation_loss

    if supervise_flow:
        assert gt_flow is not None

        flow_loss = compute_flow_loss(predicted_flow, gt_flow, mask)

        total_loss += flow_alpha * flow_loss
        loss["flow_loss"] = flow_loss

    loss["total_loss"] = total_loss

    return loss
