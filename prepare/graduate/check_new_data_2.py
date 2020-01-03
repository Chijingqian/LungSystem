import glob
import shutil
label_file = "H:/dataset/graduate_data/graduation_design/train_data/ct_draw_label/ct_draw_label/generate_txt/label.txt"
picture_pattern = "H:/dataset/graduate_data/graduation_design/train_data/ct_draw_label/ct_draw_label/verify_img/*"
new_picture_dir = "H:/dataset/graduate_data/graduation_design/train_data/ct_draw_label/ct_draw_label/new_verify_img"

if __name__ == "__main__":
    id_set = set()
    with open(label_file) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            p_id = line.split(",")[0] 
            z = line.split(",")[1]
            p_id = p_id+"_"+z
            id_set.add(p_id)
    print(id_set)
    print(len(id_set))
    input()
    file_list = glob.glob(picture_pattern) 
    for f_name in file_list:
        p_id = f_name.split("\\")[-1].split(".")[0]
        if p_id not in id_set:
            print(p_id)
        else:
            id_set.remove(p_id)
            shutil.copy(f_name,new_picture_dir)
    print(id_set)

            
            