import cv2
import glob

label_file = "H:/dataset/graduate_data/graduation_design/train_data/ct_draw_label/ct_draw_label/generate_txt/label.txt"
picture_pattern = "H:/dataset/graduate_data/graduation_design/train_data/ct_draw_label/ct_draw_label/verify_img/*"
color_dict = {
    "0":[255,0,0],
    "1":[0,255,0],
    "2":[0,0,255],
    "3":[255,255,0],
    "4":[0,255,255]
}

if __name__ == "__main__":
    label_dict = dict()
    with open(label_file) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            p_id,z_slice,x_pix,y_pix,r,c_type = line.split(",")
            print(p_id,z_slice,x_pix,y_pix,r,c_type)
            label_dict[p_id] = [int(z_slice),int(x_pix),int(y_pix),int(r),c_type]

    file_list = glob.glob(picture_pattern)
    for f_name in file_list[960:]:
        print(f_name)
        if f_name.split("\\")[-1] in ["11201_50.png","11542_70.png","11799_82.png",
                                      "11824_137.png","13148_80.png","20231_231.png",
                                      "20563_110.png","20563_130.png","20570_125.png",
                                      "20641_133.png","21140_86.png","21524_125.png",
                                      "21551_85.png","21558_89.png","22234_118.png",
                                      "23334_92.png","25983_125.png","8465_97.png",
                                      "8857_39.png"]:
            continue
        p_id = f_name.split("\\")[-1].split("_")[0]
        z_slice,x_pix,y_pix,r,c_type = label_dict[p_id]
        print(p_id,z_slice,x_pix,y_pix,r,c_type)
        img = cv2.imread(f_name, cv2.IMREAD_GRAYSCALE)
        cv2.circle(img, (x_pix,y_pix), r, color=color_dict[c_type], thickness=1, lineType=8, shift=0)
        cv2.imshow('检验圆形', img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()  
        
    

