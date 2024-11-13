import re
import os
from unidecode import unidecode

# memisahkan token kata dan punctuation pada kalimat, tetap menyimpan token redundant yang dihubungkan dengan dash( - ) seperti "kura-kura" sebagai satu token.
def pisahkan_kata(kalimat):
    tokens = re.findall(r'\w+(?:-\w+)*|[^\w\s]', kalimat)
    return tokens

# Bersihkan terminal, biar rapi aja :>
def bersihkan_terminal(array_kata):
    os.system("cls")
    print(f"[{', '.join([f'{word}:{index+1}' for index, word in enumerate(array_kata)])}]")

# fungsi untuk menguraikan data dengan format CoNLL-U secara manual melalui form input
def uraikan(kata, index, array_kata):
    bersihkan_terminal(array_kata)
    id = index
    form = kata
    
    lemma = input(f"Masukkan lemma kata {kata} : ")
    uppos = input(f"Masukkan UPPOS kata {kata} : ")
    xpos = input(f"Masukkan XPOS kata {kata} : ")
    feats = input(f"Masukkan FEATS kata {kata} : ")
    head = input(f"Masukkan index HEAD kata {kata} : ")
    deprel = input(f"Masukkan DEPREL kata {kata} : ")
    deps = input(f"Masukkan DEPS kata {kata} : ")
    misc = input(f"Masukkan MISC kata {kata} : ")
    return {
        "ID" : id,
        "FORM" : form,
        "LEMMA" : lemma,
        "UPPOS" : uppos,
        "XPOS" : xpos,
        "FEATS" : feats,
        "HEAD" : head,
        "DEPREL" : deprel,
        "DEPS" : deps,
        "MISC" : misc
    }


# Fungsi untuk menambah data dengan format CoNLL-U 
def buat_conllu_data(data_kalimat, filename, id_kalimat, kalimat, data_header):
    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"# id_kalimat = {id_kalimat}\n")
        file.write(f"# header = {data_header[0]}\n")
        file.write(f"# teks = {kalimat}\n")

        for kata in data_kalimat:
            baris = "\t".join([
                str(kata["ID"]),
                kata['FORM'],
                kata['LEMMA'],
                kata['UPPOS'],
                kata['XPOS'],
                kata['FEATS'],
                str(kata['HEAD']),
                kata['DEPREL'],
                kata['DEPS'],
                kata['MISC'],
            ])
            file.write(baris + "\n")
        file.write("\n")

# Fungsi untuk cek id dari kalimat sebagai pembeda satu kalimat dengan yang lain, file id_kalimat digunakan untuk mengingat "id" terakhir yang digunakan pada file CoNLL-U
def cek_id_kalimat(route):
    try:
        with open(route, "r") as file:
            id_kalimat_terakhir = file.read().strip()
            if id_kalimat_terakhir or id_kalimat_terakhir != "":
                return int(id_kalimat_terakhir)
            else:
                return 1
    except FileNotFoundError or IndexError:
        with open(route, "w") as file:
            file.write("0")
            return 1

# Fungsi untuk menguraikan kalimat tanpa menambah tag apa-apa ( - )
def uraikan_kosong(kata, index, array_kata):
    bersihkan_terminal(array_kata)
    id = index
    form = kata

    
    lemma = "-"
    uppos = "-"
    xpos = "-"
    feats = "-"
    head = "-"
    deprel = "-"
    deps = "-"
    misc = "-"
    return {
        "ID" : id,
        "FORM" : form,
        "LEMMA" : lemma,
        "UPPOS" : uppos,
        "XPOS" : xpos,
        "FEATS" : feats,
        "HEAD" : head,
        "DEPREL" : deprel,
        "DEPS" : deps,
        "MISC" : misc
    }


# Fungsi untuk membaca string dari data .txt, kemudian melakukan tokenisasi kalimat dengan separator " . "
def baca_txt(namafile):
    with open(namafile, "r") as file:
        konten = file.read()
    data_split = re.split(r'(?<=[^\w\s-])', konten)
    
    return [unidecode(item.strip()) for item in data_split if item.strip()]


# Fungsi untuk membaca folder dataset untuk memudahkan pembuatan data
def baca_folder(folder_path):
    array_filename = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            array_filename.append(file_path)
    return array_filename

# fungsi utama yang dipanggil untuk membuat dataset
def buat_data(folder_path, route_data, route_index):
    
    id_kalimat = cek_id_kalimat(route_index)
    array_file = baca_folder(folder_path)
    for header in array_file:
        data_cerita = baca_txt(header)
        header_text = re.findall(r'[^\\\/]+', header)
        for kalimat in data_cerita:
            array_kata = pisahkan_kata(kalimat)
            data_kalimat = []
            for index, kata in enumerate(array_kata):
                anotasi_kata_dict = uraikan_kosong(kata, index+1, array_kata)
                data_kalimat.append(anotasi_kata_dict)
            buat_conllu_data(data_kalimat, route_data, id_kalimat, kalimat, header_text[-1].split("."))
            id_kalimat += 1
            with open(route_index, "w", encoding="utf-8") as file:
                file.write(str(id_kalimat))

def buat_satu_data(namafile, route_data, route_index):
    id_kalimat = cek_id_kalimat(route_index)
    data_cerita = baca_txt(namafile)
    for kalimat in data_cerita:
        # print(kalimat)
        array_kata = pisahkan_kata(kalimat)
        data_kalimat = []
        for index, kata in enumerate(array_kata):
            anotasi_kata_dict = uraikan_kosong(kata, index+1, array_kata)
            data_kalimat.append(anotasi_kata_dict)
        buat_conllu_data(data_kalimat, route_data, id_kalimat, kalimat, (namafile.split("//"))[-1])
        id_kalimat += 1
        with open(route_index, "w", encoding="utf-8") as file:
            file.write(str(id_kalimat))


        

    
    