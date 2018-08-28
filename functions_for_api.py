import os


def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_company_folder(company_name):
    company_dir_path = './all_data/'+company_name+'/'
    temp_dir_path = './all_data/'+str(company_name)+'/Temp/'

    dataset_emp_dir_path = './all_data/' + str(company_name)+'/dataset/'

    image_dir_path = './all_data/'+str(company_name)+'/Images/'
    review_images_dir_path = './all_data'+str(company_name)+'review_images'
    create_dir_if_not_exists(company_dir_path)
    create_dir_if_not_exists(temp_dir_path)
    create_dir_if_not_exists(cropped_dir_path)
    create_dir_if_not_exists(mser_dir_path)
    create_dir_if_not_exists(Extracted_box_dir_path)
    create_dir_if_not_exists(dataset_emp_dir_path)
    create_dir_if_not_exists(img_white_dir_path)
    create_dir_if_not_exists(debugging_dir_path)
    create_dir_if_not_exists(image_dir_path)
    create_dir_if_not_exists(review_images_dir_path)


def create_temp_directory(company_name, filename, username):

    # temp = image_name.split('.')
    # del temp[-1]
    # temp_name = '.'.join(temp)
    # temp1 = image_name.split('.')
    # filename = temp_name+'_'+username+'.'+temp1[-1]

    company_dir_path = './all_data/'+company_name+'/'
    training_dir_path = './all_data/'+company_name+'/'+'Training/'
    training_emp_dir_path = './all_data/'+company_name+'/'+'Training/'+username+'/'
    training_emp_image_dir_path = './all_data/'+company_name+'/'+'Training/'+username+'/'+filename+'/'
    improving_dir_path = './all_data/'+company_name+'/'+'imporving/'
    improving_emp_dir_path = './all_data/'+company_name+'/'+'improving/'+username+'/'
    improving_emp_image_dir_path = './all_data/'+company_name+'/'+'improving/'+username+'/'+filename+'/'
    temp_dir_path = './all_data/'+str(company_name)+'/Temp/'
    temp_file_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/'
    cropped_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/cropped/'
    mser_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/mser/'
    Extracted_box_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/Extracted_box/'
    dataset_dir_path = './all_data/' + str(company_name)+'/dataset/'
    dataset_emp_dir_path = './all_data/' + str(company_name)+'/dataset/'+'dataset_emp_'+username+'/'
    img_white_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/img_white/'
    debugging_dir_path = './all_data/'+str(company_name)+'/Temp/'+filename+'/Debugging/'
    image_dir_path = './all_data/'+str(company_name)+'/Images/'
    review_images_dir_path = './all_data'+str(company_name)+'review_images'

    create_dir_if_not_exists(company_dir_path)
    create_dir_if_not_exists(training_dir_path)
    create_dir_if_not_exists(training_emp_dir_path)
    create_dir_if_not_exists(training_emp_image_dir_path)
    create_dir_if_not_exists(improving_dir_path)
    create_dir_if_not_exists(improving_emp_dir_path)
    create_dir_if_not_exists(improving_emp_image_dir_path)
    create_dir_if_not_exists(temp_dir_path)
    create_dir_if_not_exists(temp_file_dir_path)
    create_dir_if_not_exists(cropped_dir_path)
    create_dir_if_not_exists(mser_dir_path)
    create_dir_if_not_exists(Extracted_box_dir_path)
    create_dir_if_not_exists(dataset_dir_path)
    create_dir_if_not_exists(dataset_emp_dir_path)
    create_dir_if_not_exists(img_white_dir_path)
    create_dir_if_not_exists(debugging_dir_path)
    create_dir_if_not_exists(image_dir_path)
    create_dir_if_not_exists(review_images_dir_path)


def delete_entire_directory(company_name, image_name, username):
    temp = image_name.split('.')
    del temp[-1]
    temp_name = '.'.join(temp)
    temp1 = image_name.split('.')
    filename = temp_name+'_'+username+'.'+temp1[-1]

    top = './all_data/'+str(company_name)+'/Temp/'+filename+'/'

    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

    os.rmdir(top)


def create_dataset_for_emplyee_folder(company_name, employee_id):
    dataset_emp_dir_path = './all_data/' + str(company_name)+'/dataset/dataset_emp_'+employee_id
    create_dir_if_not_exists(dataset_emp_dir_path)
