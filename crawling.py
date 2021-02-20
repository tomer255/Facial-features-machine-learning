from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import os
from os import path
import urllib.request
import time
import cv2
import dlib
import numpy as np

SHUTTER_BASE_URL = r"https://www.shutterstock.com/search/"


def face_detection(link, detector, filename, dire):
    if path.exists(dire + filename): #Checking if a File Exists
        print(link, ' -- file already exists SKIP')
        return False
    try:
        resp = urllib.request.urlopen(link)
    except Exception as e:
        print(e)
        print(link, ' -- ERROR: link fail !!!')
        return False
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 1:
        print(link, ' -- face detected successfully')
        cv2.imwrite(dire + filename, image)  # seve image
        return True
    else:
        print(link, ' -- face NOT detected')
        return False


def generate_search_url(params):
    search_values = ['people', 'faces']
    url = SHUTTER_BASE_URL + "+".join(search_values) + "?"
    dict = {"mreleased": "true", "number_of_people": "1"}
    url = url_add_params(url, dict)
    url = url_add_params(url, params)
    return url[:-1]  # Remove trailing "&"


def url_add_params(url, dict):
    for k, v in dict.items():
        url += k + "=" + v + "&"  # Concat the parameters to the url.
    return url



def scroll_down(body):
    for _ in range(20):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.5)



def shutterstock_crawling():
    params = {"age": "20s", "gender": "female"}
    start_page = 95
    options = Options()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome("./chromedriver.exe", options=options)
    gender_map = {"male": "0", "female": "1"}
    age_map = {"infants": "0", "children": '1', 'teenagers': '2', '20s': '3', '30s': '4', '40s': '5',
               '50s': '4', '60s': '5', 'older': '6'}

    detector = dlib.get_frontal_face_detector()

    filtered_url = generate_search_url(params)
    url = filtered_url + '&page=' + str(start_page)
    driver.get(url)
    body = driver.find_element_by_tag_name('body')
    next_pg = driver.find_element_by_link_text('Next')

    dire = "./images/" + params['gender'] + '/' + params['age'] + '/'
    start_filename = gender_map[params['gender']] + '_' + age_map[params['age']] + '_'
    if not path.exists(dire): #Checking if a Directory Exists
        os.mkdir(dire)
    while 1:
        scroll_down(body)
        try:
            elements = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
                (By.XPATH, "//img[contains(@data-automation, 'mosaic-grid-cell-image')]")))
        except Exception as e:
            print(e)
            print(driver.current_url,'ERROR: page fail SKIP')
        else:
            for elem in elements:
                link = elem.get_attribute("src")
                filename = start_filename + link.split(sep='-')[-1]
                face_detection(link, detector, filename, dire)
                time.sleep(1)

        body.send_keys(Keys.HOME)
        time.sleep(1)
        if next_pg.get_attribute("disabled"):
            break
        next_pg.click()
        time.sleep(1)
    driver.quit()
    return 0
