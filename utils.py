import numpy as np
import json
from tqdm import tqdm



def transform_api(json_file):
    with open(json_file,"r") as f:
        dataset = json.load(f)

    t = " Bony structures include sella floor, tuberculum sella, ica prominence, clival recess, optic carotid recess, and optic prominence.\n"

    new_dataset = []
    for i in dataset:
        conv = i["conversations"]
        if "bony structure" in conv[0]["value"]:
            i["conversations"][1]["thoughts"] += t
            i["conversations"][1]["API_params"] =  {'caption': 'sella floor . tuberculum sella . ica prominence . clival recess . optic carotid recess . optic prominence'}
            new_dataset.append(i)


    with open(json_file,"w") as f:
        f.write(json.dumps(new_dataset,indent=2))


def add_caption_question(llava_558k_file_path,caption_dataset):
    with open(caption_dataset,"r") as f:
        data = json.load(f)
    with open(llava_558k_file_path,"r") as f:
        data2 = json.load(f)
    questions_list = []
    for i in data2:
        questions_list.append(i['conversations'][0]["value"])
    new_list = []
    for i in tqdm(data):
        try:
            if "idle" in i["caption"]:
                continue
            i["id"] = i["img_path"].split(".")[0]
            i["image"] = i["img_path"].split("-")[0]+"/"+i["img_path"]
            i["conversations"] = [
                {'from': 'human',
                'value': np.random.choice(questions_list)},
                {'from': 'gpt',
        'value': i['sentence']}
            ]
            new_list.append(i)
        except Exception:
            continue    
    with open(caption_dataset,"w") as f:
        f.write(json.dumps(new_list,indent=2))


def delete_img_token(json_list):
    for json_path in json_list:
        with open(json_path,"r") as f:
            dataset = json.load(f)
            for i in dataset:
                i["conversations"][2]["value"] = i["conversations"][2]["value"].replace("<image>\n","")
                i["conversations"][2]["value"] = i["conversations"][2]["value"].replace("\n<image>","")
        with open(json_path,"w") as f:
            f.write(json.dumps(dataset,indent=2))

