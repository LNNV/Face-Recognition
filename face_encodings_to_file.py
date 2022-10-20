import face_recognition
import pickle

ky_imgae = face_recognition.load_image_file("sample/Ky.jpg")
ky_face_encoding = face_recognition.face_encodings(ky_imgae)[0]

ky_up_imgae = face_recognition.load_image_file("sample/Ky_Up.jpg")
ky_up_face_encoding = face_recognition.face_encodings(ky_up_imgae)[0]

ky_down_imgae = face_recognition.load_image_file("sample/Ky_Down.jpg")
ky_down_face_encoding = face_recognition.face_encodings(ky_down_imgae)[0]

ky_left_imgae = face_recognition.load_image_file("sample/Ky_Left.jpg")
ky_left_face_encoding = face_recognition.face_encodings(ky_left_imgae)[0]

ky_right_imgae = face_recognition.load_image_file("sample/Ky_Right.jpg")
ky_right_face_encoding = face_recognition.face_encodings(ky_right_imgae)[0]

nv_imgae = face_recognition.load_image_file("sample/NV.jpg")
nv_face_encoding = face_recognition.face_encodings(nv_imgae)[0]

nv_up_imgae = face_recognition.load_image_file("sample/NV_Up.jpg")
nv_up_face_encoding = face_recognition.face_encodings(nv_up_imgae)[0]

nv_down_imgae = face_recognition.load_image_file("sample/NV_Down.jpg")
nv_down_face_encoding = face_recognition.face_encodings(nv_down_imgae)[0]

nv_left_imgae = face_recognition.load_image_file("sample/NV_Left.jpg")
nv_left_face_encoding = face_recognition.face_encodings(nv_left_imgae)[0]

nv_right_imgae = face_recognition.load_image_file("sample/NV_Right.jpg")
nv_right_face_encoding = face_recognition.face_encodings(nv_right_imgae)[0]

dv_imgae = face_recognition.load_image_file("sample/DV.jpg")
dv_face_encoding = face_recognition.face_encodings(dv_imgae)[0]

dv_up_imgae = face_recognition.load_image_file("sample/DV_Up.jpg")
dv_up_face_encoding = face_recognition.face_encodings(dv_up_imgae)[0]

dv_down_imgae = face_recognition.load_image_file("sample/DV_Down.jpg")
dv_down_face_encoding = face_recognition.face_encodings(dv_down_imgae)[0]

dv_left_imgae = face_recognition.load_image_file("sample/DV_Left.jpg")
dv_left_face_encoding = face_recognition.face_encodings(dv_left_imgae)[0]

dv_right_imgae = face_recognition.load_image_file("sample/DV_Right.jpg")
dv_right_face_encoding = face_recognition.face_encodings(dv_right_imgae)[0]

bao_imgae = face_recognition.load_image_file("sample/Bao.jpg")
bao_face_encoding = face_recognition.face_encodings(bao_imgae)[0]

bao_up_imgae = face_recognition.load_image_file("sample/Bao_Up.jpg")
bao_up_face_encoding = face_recognition.face_encodings(bao_up_imgae)[0]

bao_down_imgae = face_recognition.load_image_file("sample/Bao_Down.jpg")
bao_down_face_encoding = face_recognition.face_encodings(bao_down_imgae)[0]

bao_left_imgae = face_recognition.load_image_file("sample/Bao_Left.jpg")
bao_left_face_encoding = face_recognition.face_encodings(bao_left_imgae)[0]

bao_right_imgae = face_recognition.load_image_file("sample/Bao_Right.jpg")
bao_right_face_encoding = face_recognition.face_encodings(bao_right_imgae)[0]

thai_imgae = face_recognition.load_image_file("sample/Thai.jpg")
thai_face_encoding = face_recognition.face_encodings(thai_imgae)[0]

thai_up_imgae = face_recognition.load_image_file("sample/Thai_Up.jpg")
thai_up_face_encoding = face_recognition.face_encodings(thai_up_imgae)[0]

thai_down_imgae = face_recognition.load_image_file("sample/Thai_Down.jpg")
thai_down_face_encoding = face_recognition.face_encodings(thai_down_imgae)[0]

thai_left_imgae = face_recognition.load_image_file("sample/Thai_Left.jpg")
thai_left_face_encoding = face_recognition.face_encodings(thai_left_imgae)[0]

thai_right_imgae = face_recognition.load_image_file("sample/Thai_Right.jpg")
thai_right_face_encoding = face_recognition.face_encodings(thai_right_imgae)[0]

known_face_encodings = [
    nv_face_encoding,
    nv_up_face_encoding,
    nv_down_face_encoding,
    nv_left_face_encoding,
    nv_right_face_encoding,
    dv_face_encoding,
    dv_up_face_encoding,
    dv_down_face_encoding,
    dv_left_face_encoding,
    dv_right_face_encoding,
    bao_face_encoding,
    bao_up_face_encoding,
    bao_down_face_encoding,
    bao_left_face_encoding,
    bao_right_face_encoding,
    ky_face_encoding,
    ky_up_face_encoding,
    ky_down_face_encoding,
    ky_left_face_encoding,
    ky_right_face_encoding,
    thai_face_encoding,
    thai_up_face_encoding,
    thai_down_face_encoding,
    thai_left_face_encoding,
    thai_right_face_encoding
]

pickle.dump(known_face_encodings, open('face_encodings.txt', 'wb'))

