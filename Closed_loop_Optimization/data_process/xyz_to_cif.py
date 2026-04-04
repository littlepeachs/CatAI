
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
import os

error_counter = 0
"""
Please correct the following lines:
"""
folder_path = '/path/to/catai_data/oct_Step1/base'
save_path = '/path/to/catai_data/oct_Step1/oct_Step1_cif'
csv_path = ''
ID_name = 'mp_id'
label_name = 'cif'
"""
Stop changing from here.
"""


def read_txt_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        print("processing: ", file_path)
        # a file:
        if os.path.isfile(file_path):
            if filename.endswith('.xyz'):
                convert_xyz_to_cif(file_path, filename)
        # a folder:
        elif os.path.isdir(file_path):
            read_txt_files(file_path)


def convert_xyz_to_cif(filepath, filename):

    # 从xyz文件中读取原子坐标和元素
    save_file = os.path.join(save_path, filename[:-3] + "cif")
    # if os.path.exists(save_file):
    #     return
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
            num_atoms = int(lines[0])
            comment = lines[1].strip()
            # atoms = []
            species, coords = [], []
            for line in lines[2:]:
                words = line.split()
                # atoms.append((words[0], [float(x) for x in words[1:4]]))
                species.append(words[0])
                coords.append([float(x) for x in words[1:4]])
        # 创建Structure对象
        lattice = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 假设的晶格参数,占位符
        # print(atoms)

        # structure = Structure(lattice, atoms, coords_are_cartesian=True)
        structure = Structure(lattice, species, coords)
        # 写入cif文件
        # save_file = os.path.join(save_path, filename[:-3]+"cif")
        print("saving: ", save_file)
        CifWriter(structure).write_file(save_file)
    except Exception as e:
        # 处理其他类型的异常
        print(f"Error occurs: {e}")
        global error_counter
        error_counter += 1

def convert_from_csv():
    df = pd.read_csv(csv_path, usecols=[ID_name, label_name])
    df = df.dropna()
    for _, row in df.iterrows():
        mp_id = row[ID_name]
        cif_content = row[label_name]

        # 构建输出路径
        output_path = os.path.join(save_path, f"{mp_id}.cif")

        # 写入文件
        with open(output_path, 'w') as f:
            f.write(cif_content)
    
    
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

read_txt_files(folder_path)
print("# of failed file conversions: ", error_counter)
