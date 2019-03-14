import subprocess
import glob
import os
import sys


def get_output_size(path, width = 1024):

    command = ["ffprobe", path]
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = p.communicate()[0].decode()
    asr = res[res.find("DAR "):].split(']')[0][4:].split(':')
    try:
        asr = list(map(float, asr))
    except ValueError:
        asr = res[res.find("DAR "):].split(' ')[1].split(':')
        asr[1] = asr[1].split(",")[0]
        asr = list(map(float, asr))

    height = int(width / (asr[0]/asr[1]))
    return "{}x{}".format(width, height)


def extract_frames(src, dest, asr):
   
    print(src)
    print(asr)
    print(dest)
    
    command = ["ffmpeg", "-i", src, "-s", asr, "-qscale", "1", dest]
    subprocess.call(command)
    
if __name__ == "__main__":
    avi_path = "/export/livia/Database/AFEW"
    dest_path = "/export/livia/data/masih/AFEW/Faces"
    file_list = glob.iglob(f"{avi_path}/**/*.avi", recursive=True)
    # file_list.sort()  
    error = 0
    for f in file_list:
        try:
            aviName = f.split('/')[-1].rstrip('.avi')              
            save_path = os.path.join(dest_path, *f.split('/')[-3:-1], aviName)
            
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
                            
            print(f)
            print(save_path)
            output = "{}/{}-%3d.png".format(save_path, aviName)
            print('get aspect ratio')
            asr = get_output_size(f)         
            print('asr: ', asr,)
            print('extract frames')
            extract_frames(f, output, asr)
            print(aviName + ' done')
        except:
            error += 1
            print(aviName + ' failed')
    