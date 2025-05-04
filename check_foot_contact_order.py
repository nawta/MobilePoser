from nymeria.body_motion_provider import create_body_data_provider  
from nymeria.xsens_constants import XSensConstants 
import torch 
    
# # Create body data provider for a sequence  
# body_dp = create_body_data_provider(  
#     xdata_npz="/mnt/nas2/naoto/nymeria_dataset/data_recording_head_rwrist_lwrist_and_body_motion/20230607_s0_james_johnson_act0_e72nhq/body/xdata.npz",  
#     xdata_glb="/mnt/nas2/naoto/nymeria_dataset/data_recording_head_rwrist_lwrist_and_body_motion/20230607_s0_james_johnson_act0_e72nhq/body/xdata_blueman.glb"  # Optional  
# )  
  
# # Access all xdata.npz contents  
# xsens_data = body_dp.xsens_data

# # Get the foot_contacts array  
# foot_contacts = xsens_data[XSensConstants.k_foot_contacts]
# # Get indices of foot parts in the part_names list  
# r_foot_idx = XSensConstants.part_names.index("R_Foot")  # 17  
# r_toe_idx = XSensConstants.part_names.index("R_Toe")    # 18  
# l_foot_idx = XSensConstants.part_names.index("L_Foot")  # 21  
# l_toe_idx = XSensConstants.part_names.index("L_Toe")    # 22  
  
# # Display the order  
# print(f"Index of R_Foot in part_names: {r_foot_idx}")  
# print(f"Index of R_Toe in part_names: {r_toe_idx}")  
# print(f"Index of L_Foot in part_names: {l_foot_idx}")  
# print(f"Index of L_Toe in part_names: {l_toe_idx}")




# Alternative: Direct access using NumPy
# You can also directly access the data without using the BodyDataProvider:

import numpy as np  
  
# 左足から前に出してる
# # Load the data  # https://explorer.projectaria.com/nymeria/20230607_s0_james_johnson_act0_e72nhq?q=%22sequence_uid+%3D%3D+20230607_s0_james_johnson_act0_e72nhq%22&st=%220%22
# xsens_data = dict(np.load("/mnt/nas2/naoto/nymeria_dataset/data_recording_head_rwrist_lwrist_and_body_motion/20230607_s0_james_johnson_act0_e72nhq/body/xdata.npz"))  

# 右足から前に出してる
# # Load the data  # https://explorer.projectaria.com/nymeria/20230607_s0_james_johnson_act1_7xwm28?q=%22sequence_uid+%3D%3D+20230607_s0_james_johnson_act1_7xwm28%22&st=%220%22
# xsens_data = dict(np.load("/mnt/nas2/naoto/nymeria_dataset/data_recording_head_rwrist_lwrist_and_body_motion/20230607_s0_james_johnson_act1_7xwm28/body/xdata.npz"))  

# 右足から前に出してる
# Load the data  # https://explorer.projectaria.com/nymeria/20230607_s0_james_johnson_act2_yhbvpa?q=%22sequence_uid+%3D%3D+20230607_s0_james_johnson_act2_yhbvpa%22&st=%220%22
xsens_data = dict(np.load("/mnt/nas2/naoto/nymeria_dataset/data_recording_head_rwrist_lwrist_and_body_motion/20230607_s0_james_johnson_act2_yhbvpa/body/xdata.npz"))  



# Access foot_contacts  
foot_contacts = xsens_data[XSensConstants.k_foot_contacts]  
  
# Print information about the array  
print(f"foot_contacts shape: {foot_contacts.shape}")

for i in range(0, 1000): #range(foot_contacts.shape[0]):
    if i == 0:
        print(foot_contacts[i])
    if not (foot_contacts[i-1] == foot_contacts[i]).all():
        print(foot_contacts[i])