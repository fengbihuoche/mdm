import json
import os
from collections import defaultdict

folder_path = "/public/home/wangzy17/motion-diffusion-model/scene_generate/metrics/"


mean_dict = defaultdict(int)


file_names = os.listdir(folder_path)


for file_name in file_names:
    print(file_name)
    file_path = os.path.join(folder_path, file_name)
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    for key in ['transl_pdist_average', 'transl_std_average', 'param_pdist_average', 'param_std_average', 'marker_pdist_average', 'marker_std_average', 'non_collision_average', 'contact_average']:
        mean_dict[key+'_average'] += data[key]
for key in mean_dict:
    mean_dict[key] /= len(file_names)

output_path ="/public/home/wangzy17/motion-diffusion-model/scene_generate/metrics/output.json"
with open(output_path, 'w') as file:
    json.dump(mean_dict, file)