import re

from datetime import datetime, date

#
# dict = {0: 'success', 1: 'length is not right'}


def validate__password_strength(p):
    try:
        if (len(p) < 6 or len(p) > 30):
            x = 1
        elif not re.search("[a-z]", p):
            x = 2
        elif not re.search("[0-9]", p):
            x = 3
        elif not re.search("[A-Z]", p):
            x = 4
        # elif not re.search("[$#@]",p):
        #     x=5
        elif re.search("\s", p):
            x = 5
        else:
            x = 0

        # password=['a lowercase letter','an integer','a Capital letter','a special character out of: $#@ ','no spaces']
        password = ['length between 6 and 30 characters', 'a lowercase letter',
                    'an integer', 'a Capital letter', 'no spaces']
        check_list = [1, 2, 3, 4, 5]
        if x in check_list:
            for i in check_list:
                if x == i:
                    message = ['the password needs to have:', '1.A lowercase letter', '2. An integer', '3. A capital letter', '4. no spaces', '5. A length between 6 and 30 characters', 'It doesnt have ', password[
                        i-1]]

        else:
            message = 'valid password entered'

    except:
        x = 10
        message = 'ValueError'

    return x, message


def validate_confirm_password_match_password(password, confirm_password):

    try:
        if password == confirm_password:
            return 0
        else:
            return 1
    except:
        return 10


def validate_username(username):
    try:
        if (len(username) > 5 and len(username) < 30):
            if re.match("^[A-Za-z0-9_-]*$", username):
                x = 0
            else:
                x = 1
        else:
            x = 2

    except:
        x = 10

    return x


# some faults are it cannot recogonize
# john_o'connell@gmail.com is a valid email, but this code return false

# some alternative code: hard to understand now.
#     if re.match("^.+@([?)[a-zA-Z0-9-.]+.([a-zA-Z]{2,3}|[0-9]{1,3})(]?)$", email) != None:
#         # if re.match("$+@([?)[a-zA-Z0-9-.]+.([a-zA-Z]{2,3}|[0-9]{1,3})(]?)$", email) != None:
#
#         return True
# return False

# alternatives # '^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$', email)
def validate_email(email):
    try:
        match = ''
        if len(email) > 7:

            match = re.match(

                '^[A-Za-z0-9._%\'+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,6}$', email)
        if match != None:
            x = 0
        else:
            x = 1
    except:
        x = 10

    return x
# doesnt take care of aditya c.k dalmia has to be improved


def capitalize_name(s):
    try:
        # return ' '.join(w[:1].upper() + w[1:].lower() for w in s.split(' '))
        capitalize_name = ' '.join(w[:1].upper() + w[1:].lower() for w in b.split(' '))
        x = 0
    except:
        x = 10
    return x, capitalize_name

# format used is dd/mm/yyyy


def validate_dob_format(dob):
    try:
        if dob.find('/') == 1:
            dd, mm, yyyy = dob.split('/')
            dd = int(dd)
            mm = int(mm)
            yyyy = int(yyyy)
            if(mm == 1 or mm == 3 or mm == 5 or mm == 7 or mm == 8 or mm == 10 or mm == 12):
                max1 = 31
            elif(mm == 4 or mm == 6 or mm == 9 or mm == 11):
                max1 = 30
            elif(yyyy % 4 == 0 and yyyy % 100 != 0 or yyyy % 400 == 0):
                max1 = 29
            else:
                max1 = 28
            if(mm >= 1 and mm <= 12):
                if(dd >= 1 and dd <= max1):
                    if(yyyy > 1890):

                        x = 0

                    else:
                        x = 1
                else:
                    x = 2
            else:
                x = 3
        elif dob.find('-') == 1:
            dd, mm, yyyy = dob.split('-')
            dd = int(dd)
            mm = int(mm)
            yyyy = int(yyyy)
            if(mm == 1 or mm == 3 or mm == 5 or mm == 7 or mm == 8 or mm == 10 or mm == 12):
                max1 = 31
            elif(mm == 4 or mm == 6 or mm == 9 or mm == 11):
                max1 = 30
            elif(yyyy % 4 == 0 and yyyy % 100 != 0 or yyyy % 400 == 0):
                max1 = 29
            else:
                max1 = 28
            if(mm >= 1 and mm <= 12):
                if(dd >= 1 and dd <= max1):
                    if(yyyy > 1890):

                        x = 0

                    else:
                        x = 1
                else:
                    x = 2
            else:
                x = 3
        else:
            x = 4

    except:
        x = 10

    return x


def calculate_age(born):
    try:
        if born.find('/') == 1:
            born = ' '.join(w for w in born.split('/'))
        elif born.find('-') == 1:
            born = ' '.join(w for w in born.split('-'))
        born = datetime.strptime(born, "%d %m %Y")
        today = date.today()
        x = 0
        age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    except:
        x = 10
    return x, age

# numbers have to start from 4 and be 10 digits


def validating_phone_number(mobile_number):
    try:
        if re.match(r'[456789]\d{9}$', mobile_number):
            x = 0
        else:
            x = 1
    except:
        x = 10
    return x


def validate_pincode(pincode):
    try:
        if len(pincode) is 6:
            x = 0
        else:
            x = 1
    except:
        x = 10
    return x


def validate_employee_priority_value(priority_value):
    try:
        if priority_value >= 1 and priority_value <= 10 and isinstance(priority_value, int) == True:
            x = 0
        else:
            x = 1
    except:
        x = 10
    return x
