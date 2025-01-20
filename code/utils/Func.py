import torch
from tqdm import tqdm
import sys
sys.path.append('../')
import os
import numpy as np
from scipy import sparse as sp
from rdkit import Chem
from dgllife.utils.io import pmap
from utils.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from utils.featurizer import smiles_to_graph_tune
from multiprocessing import Pool
import torch
from dgl.data.utils import save_graphs
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import os
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import pathlib



def padding_target_feature(feature_dim,
                           max_seq_len,
                           seq2feature_dict):

    padded_target_feature_dict= {}

    for seq, feature in tqdm(seq2feature_dict.items(), total=len(seq2feature_dict)):
        seq_len = feature.shape[0]
        padded_feature = torch.zeros(max_seq_len, feature_dim)
        padded_feature[:seq_len, :] = feature
        padded_target_feature_dict[seq] = padded_feature

    return padded_target_feature_dict


### 提取分子smiles的特征
def extract_cp_feature(smiles_list, output_dir, num_workers = 4):

    graphs_save_path = os.path.join(output_dir,'cp_graphs.pkl')
    fps_save_path = os.path.join(output_dir, 'cp_fps.pt')
    mds_save_path = os.path.join(output_dir, 'cp_mds.pt')

    print('extracting graphs')
    # 将smiles转换为graph
    graphs = pmap(smiles_to_graph_tune,
                smiles_list,
                max_length = 5,
                n_virtual_nodes=2,
                n_jobs = num_workers)
    
    valid_graphs = []
    for  g in graphs:
        if g is not None:
            valid_graphs.append(g)
    save_graphs(graphs_save_path, 
                valid_graphs,
                labels = None)
    
    # 提取分子指纹
    print('extracting fingerprints')
    FP_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        FP_list.append(list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)))
    FP_arr = np.array(FP_list)
    FP_sp_mat = sp.csc_matrix(FP_arr) 
    fps = FP_sp_mat.todense().astype(np.float32)
    fps = torch.tensor(fps, dtype=torch.float32)
    torch.save(fps, fps_save_path)

    # 提取分子描述符
    print('extracting molecular descriptors')
    generator = RDKit2DNormalized()
    pool = Pool(num_workers)
    features_map = pool.imap(generator.process, smiles_list)
    arr = np.array(list(features_map))
    mds=arr[:,1:]
    mds = mds.astype(np.float32)
    mds = torch.from_numpy(np.where(np.isnan(mds), 0, mds))
    torch.save(mds, mds_save_path)
    return 'Done!'




### 过滤smiles字符串
def filter_invalid_smiles(smiles_list):
    """
    过滤无效的 SMILES。

    参数:
        smiles_list (list): 包含 SMILES 字符串的列表。
    
    返回:
        valid_smiles (list): 仅包含有效 SMILES 的列表。
        invalid_smiles (list): 无效 SMILES 的列表。
    """
    valid_smiles = []
    invalid_smiles = []
    
    for smiles in tqdm(smiles_list, total=len(smiles_list)):
        mol = Chem.MolFromSmiles(smiles)  # 尝试解析 SMILES
        if mol:  # 如果解析成功
            valid_smiles.append(smiles)
        else:  # 如果解析失败
            invalid_smiles.append(smiles)
    
    return valid_smiles, invalid_smiles


def seq2fasta(seq_list, save_dir):

    seq_records = [
        SeqRecord(Seq(seq), id=f"Target_{index + 1}", description="")  # id 表示注释行
        for index, seq in enumerate(seq_list)
    ]

    # 写入 FASTA 文件
    output_file = os.path.join(save_dir,'target_seq.fasta')
    SeqIO.write(seq_records, output_file, "fasta")



def extract_esm_feature(model_location, 
                        fasta_file, 
                        output_dir, 
                        toks_per_batch, 
                        repr_layers, 
                        device,
                        include, 
                        truncation_seq_length):
    
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()

    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )
    if torch.cuda.is_available():
        model = model.to(device=device)
        print("Transferred model to GPUs")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )
    print(f"Read {fasta_file} with {len(dataset)} sequences")

    
    return_contacts = "contacts" in include
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]

    result  = {}
    
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            toks = toks.to(device=device if torch.cuda.is_available() else "cpu", non_blocking=True)
            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
            logits = out["logits"].to(device=device if torch.cuda.is_available() else "cpu")
            representations = {
                layer: t.to(device=device) for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].clone()

            for i, label in enumerate(labels):
                # output_file = model_folder / f"{label.split('_')[0]}.pt"
                # output_file.parent.mkdir(parents=True, exist_ok=True)
                # result = {"label": label}
                truncate_len = min(truncation_seq_length, len(strs[i]))
                if "per_tok" in include:
    
                    for layer, t in representations.items():
                        result_tensor = t[i, 1:truncate_len + 1].clone()
                        result_tensor = result_tensor.detach().cpu()
                        result[label] = result_tensor

                if "mean" in include:

                    for layer, t in representations.items():
                        result_tensor = t[i, 1:truncate_len + 1].mean(0).clone()
                        result_tensor = result_tensor.detach().cpu()
                        result[label] = result_tensor


                if "bos" in include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[i, :truncate_len, :truncate_len].clone()

    output_path = os.path.join(output_dir, f'esm_feature.pt')
    torch.save(result, output_path)

    return 'Done!'









