import torch



if __name__=='__main__':
    # path1 = 'logs/a100_vcg_tc_ccm_blip_7kw_taiyi_traineocoder_lr5e-5_eps1e-4/Imagen/1.450000.pt'
    path1 = 'logs/main_torch/2.ckpt'
    # path2 = 'logs/mask_loss_human_enhance/Imagen/0.10000.pt'
    path2 = 'logs/main_torch/9.ckpt'

    saved_model1 = torch.load(path1, map_location="cpu")
    saved_model2 = torch.load(path2, map_location='cpu')
    # print(type(saved_model1))
    diff_dict = {}
    zeros = []
    with open("./tmp/pokemon.txt",'w') as wf:
        for k,v in saved_model1.items():
            try:
                res = v.equal(saved_model2[k])
                wf.write(f"{k}\t{res}\n")
                
            except Exception as e :
                # rr = k
                # print(k)
                print(e)