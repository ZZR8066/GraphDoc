import os


def get_filepaths(file_dir, ext='.pdf'):
    '''
        goal: 提取当前文件夹及其子文件夹下的所有'.ext'文件
        param: file_dir, 需要提取的文件夹路径
        param: ext, 需要提取的文件类型
        output: 文件类型为ext的所有文件路径
    '''
    all_files = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[-1].lower() == ext:
                all_files.append(root+"/"+file)
    return all_files


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists
