Set up thí nghiệm
Clone github https://github.com/Etdihatthoc/VN_SIGN 
Tải dataset Blur_video từ Drive https://drive.google.com/drive/folders/11CDJZo111eC36E6FalRkBBw76LI9oa1p lưu vào folder data: data/Blur_video
Tạo môi trường ảo, install các package trong requirements.txt : 
pip install -r requirements.txt
Tại tmux, active môi trường, setup bằng lệnh:
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -vE '/usr/local/cuda-11.0/lib64|/usr/local/cuda/extras/CUPTI/lib64' | tr '\n' ':')
export PATH="/usr/local/cuda-11.4/bin:$PATH" 
export LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"

Train model from scratch
ID3 one view
Train: 
python main.py --config configs/i3d/label_1_1000/i3d_one_view_from_scratch.yaml
Test 
python main.py --config configs/i3d/test_cfg/label_1_1000/i3d_one_view_from_scratch.yaml

MVIT_v2 one view
Train:
python main.py --config configs/mvit_v2/label_1_1000/mvit_v2_S_one_view_1_1000_finetune_from_scratch.yaml
Test:
python main.py --config
configs/mvit_v2/test_cfg/label_1_1000/mvit_v2_S_one_view_finetune_from_scratch.yaml

Swin Transformer one view
Train: 
python main.py --config configs/swin_transformer_3d/label_1_1000/swin_transformer_3d_T_one_view_from_scratch.yaml
Test 
python main.py --config configs/swin_transformer_3d/test_cfg/label_1_1000/swin_transformer_3d_T_one_view_from_scratch.yaml

ID3 three view
Train: 
python main.py --config configs/i3d/label_1_1000/i3d_three_view_finetune_from_one_view.yaml
Test 
python main.py --config configs/i3d/test_cfg/label_1_1000/i3d_three_view_finetune_from_one_view.yaml

MVIT_v2 three view
Train:
python main.py --config configs/mvit_v2/label_1_1000/mvit_v2_S_three_view_1_1000_finetune_from_one_view.yaml
Test:
python main.py --config
configs/mvit_v2/test_cfg/label_1_1000/mvit_v2_S_three_view_finetune_from_one_view.yaml

Swin Transformer three view
Train: 
python main.py --config configs/swin_transformer_3d/label_1_1000/swin_transformer_3d_T_three_view_finetune_from_one_view.yaml
Test 
python main.py --config configs/swin_transformer_3d/test_cfg/label_1_1000/swin_transformer_3d_T_three_view_finetune_from_one_view.yaml
