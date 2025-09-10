# **AI Based Disaster Damage Assessment Using Pre-disaster and Post disaster Satellite images**

## **1. Problem Statement**
Disasters such as earthquakes, floods, and hurricanes cause significant damage to infrastructure, requiring rapid and accurate damage assessments. Traditional manual inspection methods are slow, hazardous, and often unreliable.  

This project aims to automate **building damage assessment** using deep learning models, leveraging **multimodal satellite imagery** to classify affected areas accurately.

## **2. Solution**
The project implements a **dual SegFormer encoder with channel attention fusion** to extract features from both **pre-disaster optical imagery** and **post-disaster SAR imagery**.  

The model classifies affected regions into four categories:
- **Background**
- **Intact**
- **Damaged**
- **Destroyed**

This deep learning solution supports emergency response by:
- **Identifying unsafe structures**
- **Prioritizing evacuation efforts**
- **Aiding resource allocation for disaster response**

## **3. Network Architecture**
The model is built using a **dual SegFormer encoder** and consists of:
- **Two SegFormer encoders** (one for optical imagery, one for SAR imagery)
- **Channel Attention Fusion** for feature extraction
- **SegFormer decoder** for generating segmentation maps
- ![Pre-Disaster](http://github.com/jatinsahu1708/AI-Based-Disaster-Damage-Assessment-Using-Pre-disaster-and-Post-disaster-Satellite-images/raw/main/Screenshot%202025-03-16%20215029.png)

This architecture improves multimodal image fusion, leading to more accurate classification.

## **4. Hyperparameters and Training Details**
- **Dataset**: BRIGHT (4,538 multimodal image pairs from 12 disaster events)
- **Input Setup**: Pre and Post Disaster Image
- **Epochs**: 35
- **Loss Functions**:
  - Dice + Focal Loss
  - Lovász + Weighted Cross-Entropy (WCE) Loss
- **Final Loss Values**:
  - **Dice + Focal Loss**: `0.0218`
  - **Lovász + WCE Loss**: `0.2`

## **5. Results**
The model achieved:
- **mIoU = 0.6322** with **Lovász + WCE Loss**
- **mIoU = 0.62** with **Dice + Focal Loss**

### **Result Visualization**
Below are some segmentation results comparing predicted and ground truth images:

| Pre-Disaster Image | Post-Disaster Image | Predicted Segmentation | Ground Truth |
|--------------------|--------------------|------------------------|-------------|
| ![Pre-Disaster](https://github.com/jatinsahu1708/AI-Based-Disaster-Damage-Assessment-Using-Pre-disaster-and-Post-disaster-Satellite-images/raw/main/la_palma-volcano_00000279_pre_disaster.png)| ![Post-Disaster](https://github.com/jatinsahu1708/AI-Based-Disaster-Damage-Assessment-Using-Pre-disaster-and-Post-disaster-Satellite-images/raw/main/la_palma-volcano_00000279_post_disaster.png) | ![Predicted](https://github.com/jatinsahu1708/AI-Based-Disaster-Damage-Assessment-Using-Pre-disaster-and-Post-disaster-Satellite-images/raw/main/output_pred_mask_colored_1.png)| ![Ground Truth](https://github.com/jatinsahu1708/AI-Based-Disaster-Damage-Assessment-Using-Pre-disaster-and-Post-disaster-Satellite-images/raw/main/output_gt_mask_colored_1.png) |

## **6. Evaluation Metrics**
The following metric was used to evaluate the model:
- **Mean Intersection over Union (mIoU)**
  - **mIoU with Lovász + WCE Loss**: `0.6322`
  - **mIoU with Dice + Focal Loss**: `0.62`

## **8. Future Improvements**
- Enhance **generalization** across different disaster types
- Implement **self-supervised learning** for better feature extraction
- Optimize **inference speed** for real-time deployment

## **9. References**
- **Chen, Y., Li, J., & Wang, X.** "BRIGHT: A Globally Distributed Multimodal Building Damage Assessment Dataset." *arXiv preprint*, 2024.
- **Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J. M., & Luo, P.** "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." *NeurIPS*, 2021.
- **Guo, J., Huang, X., & Lin, L.** "Channel-Wise Attention Mechanisms for Feature Selection in Segmentation." *IEEE Transactions on Image Processing*, 2021.

---

🚀 **This project integrates cutting-edge deep learning techniques for disaster damage assessment, aiming to improve disaster response and recovery efforts.**  


