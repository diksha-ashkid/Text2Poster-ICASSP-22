import torch
import cv2
import os
import numpy as np
from model.distrib_model import LayoutsDistribModel
from utils.anchor_utils import get_candidates_region, get_text_region
import pandas as pd

scale_val = 20
channel_deep = 16
position_deep = 8
STD_WIDTH, STD_HEIGHT = 300, 400
channel_deep = 16
MIN_VALUE = -999999
MAX_BBOX_NUM = 2



distrib_model = LayoutsDistribModel(
    dim_feedforward=channel_deep,
    scale_val=scale_val,
    channel_deep=channel_deep,
    position_deep=position_deep)

ckpt_path = "./checkpoint/27.80619_distribCNN_BigPosition_epoch_76_scale_20.pth"
distrib_model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')).module.state_dict())

saliency = cv2.saliency.StaticSaliencyFineGrained_create()

def softmax_1d_weight(x, weight=1):
    exp_x = np.exp(x * weight)
    exp_sum = exp_x.sum()
    return exp_x / exp_sum * x.shape[0]


def smooth_region_dectection(img):
    (success, saliency_map) = saliency.computeSaliency(img)
    scaled_saliency_map = cv2.resize(saliency_map, (STD_WIDTH, STD_HEIGHT))
    smooth_regions, smooth_scores = get_candidates_region(scaled_saliency_map, threshold=2)
    regions = np.array(
        [[[obj[0], obj[1]],
          [obj[2], obj[1]],
          [obj[2], obj[3]],
          [obj[0], obj[3]]] for obj in smooth_regions.numpy()],
        dtype=np.int32) // 1
    smooth_region_mask = np.zeros(shape=(1, STD_HEIGHT, STD_WIDTH), dtype=np.uint8)
    cv2.fillPoly(smooth_region_mask[0], regions, 1)
    return smooth_region_mask, regions, saliency_map


def get_distrib_mask(cand_mask):
    input_mask = torch.tensor(cand_mask).float()

    with torch.no_grad():
        pred_decoder_bbox_map, _ = distrib_model.forward(inputs_candidates_masks=input_mask,
                                                         outputs_bboxes_masks=None, extract=True)
    decoder_bbox_map = pred_decoder_bbox_map.clone().cpu().numpy()[0][0]
    decoder_bbox_map = cv2.resize(decoder_bbox_map, (STD_WIDTH, STD_HEIGHT))
    return decoder_bbox_map


def get_bbox_mask(bbox):
    mask = np.zeros((1, STD_HEIGHT, STD_WIDTH), dtype=np.uint8)
    regions = np.array([[[obj[0], obj[1]],
                         [obj[2], obj[1]],
                         [obj[2], obj[3]],
                         [obj[0], obj[3]]] for obj in bbox[0][:min(MAX_BBOX_NUM, data_len)]], dtype=np.int32)
    cv2.fillPoly(mask[0], regions, 1, 1)
    return mask


if __name__ == "__main__":
    img_path = r"D:\Personal\diksha\AS\Text2Poster-ICASSP-22\bk_image_folder\quote.png"
    img = cv2.imread(img_path)
    width, height = img.shape[1], img.shape[0]
    img_size = (width, height)


    
    smooth_region_mask, regions, saliency_map = smooth_region_dectection(img)
    bbox_distrib_map = get_distrib_mask(smooth_region_mask)

    cv2.imwrite("./display/candidate_regions.jpg", smooth_region_mask[0] * 255)
    cv2.imwrite("./display/layout_distribution.jpg", bbox_distrib_map * 255)

    
# Prepare the data for DataFrame
    data = {'Region': [], 'Coordinates': [], 'Area': []}

# Calculate the area and coordinates for each region
    for i, region in enumerate(regions):
        # Calculate the bounding rectangle coordinates
        x, y, w, h = cv2.boundingRect(region)

        # Calculate the area of the region
        area = cv2.contourArea(region)

        # Append the data to the dictionary
        data['Region'].append(i+1)
        data['Coordinates'].append(f"({x}, {y}, {x+w}, {y+h})")
        data['Area'].append(area)

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Export the DataFrame to an Excel file
    df.to_excel('./display/smooth_region_coordinates.xlsx', index=False)

    # Show (salicy_map, smooth_region) in a figure.
    saliency_map_with_smooth = np.zeros((height, width, 3))
    saliency_map_with_smooth[:, :, 0] = saliency_map / saliency_map.max()
    smooth_region_mask = cv2.resize(smooth_region_mask[0], (width, height))
    saliency_map_with_smooth[:, :, 2] = smooth_region_mask / smooth_region_mask.max()
    saliency_map_with_smooth = cv2.resize(saliency_map_with_smooth, (width, height))
    cv2.imwrite("./display/saliency_map_with-smooth-pw-13.jpg", saliency_map_with_smooth * 255)

    # Show (salicy_map, predicted_layout_distribution) in a figure.
    saliency_map_with_distrib = np.zeros((height, width, 3))
    saliency_map_with_distrib[:, :, 0] = saliency_map / saliency_map.max()
    bbox_distrib_map = cv2.resize(bbox_distrib_map, (width, height))
    saliency_map_with_distrib[:, :, 2] = bbox_distrib_map / bbox_distrib_map.max()
    saliency_map_with_distrib = cv2.resize(saliency_map_with_distrib, (width, height))
    cv2.imwrite("./display/ppt_bkg.jpg", saliency_map_with_distrib * 255)
