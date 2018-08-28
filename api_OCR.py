from flask import Flask, render_template, url_for, request, session, redirect, jsonify, send_from_directory

from flask_pymongo import PyMongo

import bcrypt
import re
import ocr_segmentation
import functions_for_login
import blur_detection
import os
import functions_for_api


# response_code=100   is success
# response_code=101 wrong details
# response_code=102
# response_code=404 exception error/ValueError
#
#
#
app = Flask(__name__)

app = Flask(__name__, static_folder='all_data', static_url_path='/all_data')
app.config['MONGO_DBNAME'] = 'Marble_testing'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/Marble_testing'

#
# app.config['MONGO_DBNAME'] = 'marble_industry'
# app.config['MONGO_URI'] = 'mongodb://aditya:aditya1@ds255970.mlab.com:55970/marble_industry'


mongo = PyMongo(app)

UPLOAD_for_setting_server = '192.168.43.119:61000'
UPLOADS = UPLOAD_for_setting_server

# UPLOAD_for_setting_server = '192.168.0.107:61000'
# UPLOADS = UPLOAD_for_setting_server


@app.route('/')
def index():

    dict = {'response_code': 100, 'message': 'hello world'}
    return (jsonify(dict))


@app.route('/login_user')
def login_user():
    res = {}
    if session['username']:
        user = session['username']

        res['username'] = user
        res['response_code'] = 100
        res['message'] = 'success'
    else:
        res['response_code'] = 101
        res['message'] = 'internal error'
    return (jsonify(res))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/login', methods=['POST'])
def login():
    res = {}
    users = mongo.db.employee
    username = request.form['username']
    password = request.form['password']
    fr_password, fr_message = functions_for_login.validate__password_strength(password)
    fr_username = functions_for_login.validate_username(username)
    try:
        if fr_username is 0:
            login_user = users.find_one({'username': username})

            if login_user:
                if fr_password is 0:
                    # a = bcrypt.hashpw(password.encode('utf-8'),
                    #                   password.encode('utf-8'))
                    # b=login_user['password'].encode('utf-8')
                    if password == login_user['password']:
                        session['username'] = username
                        # return redirect(url_for('index'))
                        res['response_code'] = 100
                        res['message'] = 'login success'

                    else:
                        res['response_code'] = 101
                        res['message'] = 'wrong password'
                        res['fr_password'] = fr_password

                elif fr_password == 10:
                    res['response_code'] = 404
                    res['message'] = 'function response error'
                    res['fr_password'] = fr_password
                    res['fr_password_message'] = 'try, except error in validating basic password parameters'

                else:

                    res['response_code'] = 101
                    res['message'] = 'wrong password'
                    res['fr_password'] = fr_password

            else:
                res['response_code'] = 102
                res['message'] = 'user not found'

        elif fr_username == 10:
            res['response_code'] = 404
            res['message'] = 'function response error'
            res['fr_usernanme'] = fr_username
            res['fr_username_messgae'] = 'try, except error in validating basic username parameters'

        else:
            res['response_code'] = 101
            res['message'] = 'wrong username'
            res['fr_username'] = fr_username

    except:
        res['response_code'] = 500
        res['message'] = 'server error'

    return (jsonify(res))


@app.route('/register', methods=['POST'])
def register():
    res = {}
    users = mongo.db.employee
    if request.method == 'POST':

        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        contact_number = request.form['contact_number']
        company = request.form['company_name']
        username = request.form['username']
        existing_user = users.find_one({'username': username})

        # res_func_email = functions_for_login.validate_email(email)
        # res_func_username = functions_for_login.validate_username(username)
        # res_func_contact_number = functions_for_login.validating_phone_number(contact_number)

        # res_func_password = functions_for_login.validate__password_strength(password)
        # res_func_confirm_password = functions_for_login.validate_confirm_password_match_password(
        #     password, confirm_password)
        # if res_func_email == 0 and res_func_password == 0 and res_func_username == 0 and res_func_confirm_password == 0:
        if existing_user is None:
            # hashpass=bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt())
            users.insert({'username': username, 'password': password, 'name': name,
                          'email': email, 'contact_number': contact_number, 'company': company})
            session['username'] = request.form['username']
            res['response_code'] = 100
            res['message'] = 'user successfully added and session created for following user'

        else:
            res['response_code'] = 1101
            res['message'] = 'username already taken'
        # else:
        #     res['response_code'] = 404
        #     res['message'] = 'Some detail do notmatch the minimum criteria'
    else:
        res['response_code'] = 404
        res['message'] = 'unknown error'

    return (jsonify(res))


@app.route('/upload_image/', methods=['POST'])
def upload_image():

    res = {}
    if not request.method == 'POST':
        return jsonify(**method_not_allowed)

    x = request.files['fileToUpload']
    username = request.form['username']
    company_name = request.form['company_name']

    # filename = str(int(time.time())) + '_' + x.filename
    temp = x.filename.split('.')
    del temp[-1]
    temp_name = '.'.join(temp)
    temp1 = x.filename.split('.')
    filename = temp_name+'_'+username+'.'+temp1[-1]

    file_path = './all_data/'+str(company_name)+'/Images/'
    file_path_to_display = file_path[1:]
    if os.path.exists(file_path):
        x.save(file_path+filename)
        image_path = UPLOADS + file_path+filename
        image_path_to_display = UPLOADS+file_path_to_display+filename

        res['response_code'] = 100
        res['message'] = 'uploaded successfully'
        res['image_path'] = image_path_to_display
    elif not os.path.exists('./'+str(company_name)+'/'):
        res['response_code'] = 404
        res['message'] = 'A directory for this company is missing'

    elif not os.path.exists('./'+str(company_name)+'/Images/'):
        os.makedirs(file_path)
        x.save(file_path+filename)

        image_path_to_display = UPLOADS+file_path_to_display+filename

        res['response_code'] = 100
        res['message'] = 'uploaded successfully and directory of imagesfor company created'
        res['image_path'] = image_path_to_display

    text, fm, message, image_path = blur_detection.run_blur_detection(company_name, filename)
    res['fm_value'] = fm
    if fm > 50:
        res['blurred_image'] = 'NO'
        res['blur_message'] = message

        review_image_path, review_image_checker = ocr_segmentation.create_review_image(
            company_name, filename)
        if review_image_checker == 1:
            res['review_image_path'] = UPLOADS+review_image_path[1:]
            res['review_image_checker'] = 1
        elif review_image_checker == 0:
            res['review_image_path'] = 'contours could not be found'
            res['review_image_checker'] = 0

    else:
        res['blurred_image'] = 'YES'
        res['review_image_path'] = '0'
    res['image_name'] = x.filename

    return (jsonify(res))


@app.route('/run_OCR', methods=['GET'])
def run_ml():
    if request.method == 'GET':
        print('start')
        del_dir = request.args.get('del_dir')
        username = request.args.get('username')
        company_name = request.args.get('company_name')
        image_name = request.args.get('image_name')

        temp = image_name.split('.')
        del temp[-1]
        temp_name = '.'.join(temp)
        temp1 = image_name.split('.')
        filename = temp_name+'_'+username+'.'+temp1[-1]

        dir_path = './all_data/'+str(company_name)+'/Images/'
        file_path = dir_path+filename
        output = ocr_segmentation.Run_OCR(company_name, username, image_name)

        out = {}
        res = {}
        list_of_single_value_entries = []
        total_area = 0
        i = 1
        number_of_entries = len(output)
        for entries in output:
            # code doesnt work if line 160 is missing even though it is initialised outside on line 153
            res = {}
            res['entry_number'] = entries[0]
            res['length'] = entries[1]
            res['width'] = entries[2]
            res['area'] = entries[3]
            res['copied_above_value'] = int(entries[4])
            total_area = total_area+entries[3]
            # res['entry_number']
            # print(res)
            # print(i)
            # print('\n')
            # print('\n')
            # print('\n')

            list_of_single_value_entries.append(res)
            # print(list_of_single_value_entries)
            # print('\n')
            # print('\n')
            # print('\n')
        out['response_code'] = 100
        out['response_code'] = 'succsfull converted and added to database'
        out['single_value_entries'] = list_of_single_value_entries
        out['total_area'] = total_area

        employee = mongo.db.employee
        image_data = mongo.db.image_data
        X = {'uploader_username': username, 'image_path': UPLOADS+file_path[1:], 'company_name': company_name, 'image_name': image_name, 'image_server_name': filename,
             'single_entry_values': list_of_single_value_entries, 'total_area': total_area}

        image_data.insert(X)
        if int(del_dir) == 1:
            functions_for_api.delete_entire_directory(company_name, image_name, username)
        else:
            pass
        return (jsonify(out))


@app.route('/generate_pre_training_data', methods=['GET'])
def run_ml_1():
    if request.method == 'GET':

        username = request.args.get('username')
        company_name = request.args.get('company_name')
        image_name = request.args.get('image_name')
        ocr_segmentation.training_data_initial(company_name, username, image_name)


@app.route('/get_pre_training_data', methods=['GET'])
def run_ml_2():
    if request.method == 'GET':

        username = request.args.get('username')
        company_name = request.args.get('company_name')

        training_emp_dir_path = './all_data/'+company_name+'/'+'Training/'+username+'/'

        training_emp_dir_path_list = os.listdir(training_emp_dir_path)
        res = {}
        out = []
        for dir in training_emp_dir_path_list:
            temp = {}
            temp['image_server_name'] = dir
            training_images = os.listdir(training_emp_dir_path+dir+'/')

            list = []
            idx = 1
            for training_image in training_images:
                temp1 = {}
                temp1['training_image_path'] = UPLOADS + \
                    training_emp_dir_path[1:]+dir+'/'+training_image
                temp1['training_image_index'] = idx
                list.append(temp1)
                idx = idx+1
            temp['all_training_images'] = list
            out.append(temp)
        res['response_code'] = 100
        res['response_code'] = 'succsfull converted and added to database'
        res['all_data'] = out

        return (jsonify(res))


@app.route('/get_all_uploads_by_username')
def get_all_uploads_by_username():

    username = request.args.get('username')

    employee = mongo.db.employee
    image_data = mongo.db.image_data
    # if username == session['username']:
    res = {}
    single_image_data = []
    i = 1
    employee = mongo.db.employee
    image_data = mongo.db.image_data
    for image_data in image_data.find({"uploader_username": username}):
        # image_data['upload_number'] = i
        # print(image_data)
        i = i+1

        # format_obj_id = image_data.get('_id')
        # str_obj_id = str(format_obj_id)
        # print(str_obj_id)
        #
        # image_data['_id'] = str_obj_id
        # print(image_data)
        # if image_data.get('_id'):
        format_obj_id = image_data.get('_id')
        str_obj_id = str(format_obj_id)
        print(str_obj_id)

        image_data['_id'] = str_obj_id
        print(image_data)
        single_image_data.append(image_data)
        # print(single_image_data)
    res['response_code'] = 100
    res['response_messgae'] = 'success'
    res['all_images_uploaded_by_username'] = single_image_data

    return (jsonify(res))


@app.route('/get_all_uploads_by_company')
def get_all_uploads_by_company_name():

    company_name = request.args.get('company_name')

    employee = mongo.db.employee
    image_data = mongo.db.image_data
    # if username == session['username']:
    res = {}
    single_image_data = []
    i = 1
    employee = mongo.db.employee
    image_data = mongo.db.image_data
    for image_data in image_data.find({"company_name": company_name}):
        # image_data['upload_number'] = i
        # print(image_data)
        i = i+1

        # format_obj_id = image_data.get('_id')
        # str_obj_id = str(format_obj_id)
        # print(str_obj_id)
        #
        # image_data['_id'] = str_obj_id
        # print(image_data)
        # if image_data.get('_id'):
        format_obj_id = image_data.get('_id')
        str_obj_id = str(format_obj_id)
        print(str_obj_id)

        image_data['_id'] = str_obj_id
        print(image_data)
        single_image_data.append(image_data)
        # print(single_image_data)
    res['response_code'] = 100
    res['response_messgae'] = 'success'
    res['all_images_uploaded_by_company'] = single_image_data

    return (jsonify(res))


@app.route('/get_single_upload')
def get_single_upload_by_username():

    username = request.args.get('username')
    image_name = request.args.get('image_name')
    company_name = request.args.get('comapny_name')
    temp = image_name.split('.')
    del temp[-1]
    temp_name = '.'.join(temp)
    temp1 = image_name.split('.')
    filename = temp_name+'_'+username+'.'+temp1[-1]

    dir_path = './all_data/'+str(company_name)+'/Images/'
    file_path = dir_path+filename
    employee = mongo.db.employee
    image_data = mongo.db.image_data
    path = './all_images/' + image_name
    # if username == session['username']:
    res = {}
    single_image_data = []
    i = 1
    employee = mongo.db.employee
    image_data = mongo.db.image_data
    for image_data in image_data.find({"uploader_username": username, "image_server_name": filename}):
        # image_data['upload_number'] = i
        # print(image_data)
        # image_data = image_data[1]
        i = i+1
        if image_data.get('_id'):
            format_obj_id = image_data.get('_id')
            str_obj_id = str(format_obj_id)
            print(str_obj_id)

            image_data['_id'] = str_obj_id
            # print(image_data)
        single_image_data.append(image_data)

    res['response_code'] = 100
    res['response_messgae'] = 'success'
    res['all_images_uploaded_by_user_data'] = single_image_data

    return (jsonify(res))


import socket
from contextlib import closing


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


port = UPLOAD_for_setting_server.split(':')[1]
ip = UPLOAD_for_setting_server.split(':')[0]
if __name__ == '__main__':
    app.secret_key = 'mysecret'
    app.run(debug=True, host=ip, port=port)
    # app.run(debug=True, port=60169)
