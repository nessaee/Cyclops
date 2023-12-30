

import os, zipfile, shutil

dir_name = os.path.join(os.getcwd(),'one_lane')
extension = ".zip"
out_name = os.getcwd()
def unzip(dir_name):
    os.chdir(dir_name) # change directory from working dir to dir with files
    counter = 0
    iter = 0
    for item in os.listdir(dir_name): # loop through items in dir
        if item.endswith(extension): # check for ".zip" extension
            file_name = os.path.abspath(item) # get full path of files
            res = list(filter(lambda x: x.isdigit(), file_name.split()))
     
            zip_ref = zipfile.ZipFile(file_name) # create zipfile object
            zip_ref.extractall(dir_name+"/"+str(counter)) # extract file to dir
            zip_ref.close() # close file
            iter += 1
            if iter % 2 == 0: counter += 1
            os.remove(file_name) # delete zipped file
            
def collect(dir_name):
    counter = 1
    for item in os.listdir(dir_name): # loop through items in dir
        print(item)
        for vehicle in os.listdir(os.path.join(dir_name, item)):
            for label in os.listdir(dir_name+"/"+item+"/"+vehicle):
                dst = str(counter*100) + ".txt"
                print(dst)
                if "ver" in vehicle:
                    shutil.copy(dir_name+"/"+item+"/"+vehicle+"/"+label, os.path.join(out_name,"verifier/"+dst))
                else:
                    shutil.copy(dir_name+"/"+item+"/"+vehicle+"/"+label, os.path.join(out_name,"candidate/"+dst))
                counter += 1
collect(dir_name)