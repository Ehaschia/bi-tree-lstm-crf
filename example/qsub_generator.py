import os
prefix = '#!/usr/bin/env bash\n'

node_id = 14
root_path = '/home/ehaschia/Code/bi-tree-lstm-crf'
pbs_dir = 'pbs_dir'
pbs_name = 'n' + str(node_id)
rum_file_name = 'script/run0.sh'

pbs_data = "#PBS -l walltime=1000:00:00 \n#PBS -N node13 \n#PBS -l nodes=sist-gpu" + str(node_id) + \
           ":ppn=1 \n#PBS -S /bin/bash \n#PBS -k oe \n#PBS -q sist-tukw \n#PBS -u zhanglw"

# aim_dir = '/public/sist/home/zhanglw/code/sentiment/bi-tree-lstm-crf'
aim_dir = '/home/ehaschia/Code/bi-tree-lstm-crf'

if not os.path.exists(root_path + '/' + pbs_dir):
    os.makedirs(root_path + '/' + pbs_dir)

content = prefix + pbs_data + '\ncd ' + aim_dir
content += ' \nsource activate allen'
content += '\nsh ' + rum_file_name

with open(root_path + '/' + pbs_dir + '/' + pbs_name, 'w') as f:
    f.write(content)
print('generate ' + root_path + '/' + pbs_dir + '/' + pbs_name + ' Successful!')
