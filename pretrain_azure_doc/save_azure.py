import os
import shutil

def save_md_files(directory, output_directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                relative_path = os.path.relpath(root, directory)
                parent_folders = relative_path.split(os.path.sep)
                parent_folders.append(file)
                if parent_folders[0] == '.':
                    parent_folders[0] = os.path.basename(directory)
                new_filename = '_'.join(parent_folders)
                old_filepath = os.path.join(root, file)
                new_filepath = os.path.join(output_directory, new_filename)
                os.makedirs(output_directory, exist_ok=True)
                shutil.copy2(old_filepath, new_filepath)

def check_md_files(directory):
    n = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                n+=1
    return n


def save_to_plain_md(output_folder_path = './data/', read_folder_path = './repos/'):
    subfolders = [f for f in os.listdir(read_folder_path) if os.path.isdir(os.path.join(read_folder_path, f))]

    for subfolder in subfolders:             
        sub_read_folder_path = os.path.join(read_folder_path, subfolder)
        sub_save_folder_path= os.path.join(output_folder_path, subfolder.rstrip('.git'))    
        save_md_files(sub_read_folder_path, sub_save_folder_path)

if __name__ == '__main__':
    save_to_plain_md()

