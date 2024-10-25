import glob
import os
import json

dataset_name = "hagrid_dataset_512_ultrasmall_all_classes"
target_dir = "data\\" + dataset_name + "\\"

reformat_dict = {
    "mute": "Mute. Sign with index finger close to mouth",
    "ok": "OK. OK sign with thumb and index finger in circle, other fingers separated",
    "none": "None. No exact sign was shown to the camera",
    "call": "Call. Sign of imitating phone with a hand",
    "like": "Like. Sign with fist and thumb up",
    "dislike": "Dislike. Sign with fist and thumb down",
    "fist": "Fist. Sign with fist",
    "four": "Four fingers up. Sign with four fingers split and one finger pressed to palm",
    "one": "One fingers up. Sign with index finger up",
    "palm": "Palm. Sign with all finger split",
    "peace": "Peace. Sign with middle and index fingers up, other fingers as fist",
    "peace_inverted": "Peace inverted. Back of the hand shown to camera. Sign with middle and index fingers up, other fingers as fist",
    "rock": "Rock. Sign with pinkie and index fingers up, other fingers as fist",
    "stop": "Stop. Sign with all fingers up, not split",
    "stop_inverted": "Stop inverted. Back of the hand shown to camera. Sign with all fingers up, not split",
    "three": "Three fingers up. Sign with three fingers(index, middle and ring) up and split, other fingers as fist",
    "three2": "Three fingers up with thumb. Sign with three fingers(thumb, index and middle) up and split, other fingers as fist",
    "two_up": "Two fingers up together. Sign with two fingers(index and middle) up and together, other fingers as fist",
    "two_up_inverted": "Two fingers up together inverted. Back of the hand shown to camera. Sign with two fingers(index and middle) up and together, other fingers as fist",
}

prev_dir = os.getcwd()
os.chdir(target_dir)

sign_list = glob.glob("*")
dataset_list = []
for sign in sign_list:
    for filename in glob.glob(sign + "/*"):
        d = {
            "messages": [
                {"content": "<image>What sign is shown to the camera?", "role": "user"},
                {"content": reformat_dict[sign], "role": "assistant"}
            ],
            "images": [
                ("/mnt/f" + prev_dir[2:] + "\\" + target_dir + filename).replace("\\", "/")
            ]
        }
        dataset_list.append(d)

os.chdir(prev_dir)


dataset_info = {
    "hagrid": {
        "file_name": "dataset_metadata.json",
        "formatting": "sharegpt",
        "columns": {
          "messages": "messages",
          "images": "images"
        },
        "tags": {
          "role_tag": "role",
          "content_tag": "content",
          "user_tag": "user",
          "assistant_tag": "assistant"
        }
    }
}

with open(target_dir + "dataset_metadata.json", "w+") as f:
    json.dump(dataset_list, fp=f, indent=4)

with open(target_dir + "dataset_info.json", "w+") as f:
    json.dump(dataset_info, fp=f, indent=4)
