REM install virtualenv if needed
REM pip3 install virtualenv
REM virtualenv --system-site-packages ./venv
REM call .\venv\Scripts\activate
REM pip install --upgrade pip
REM pip install --upgrade tensorflow==1.15 tqdm matplotlib Pillow imageio

REM ----- smoke_pos_size, 2D
REM 1. generate dataset
REM ..\manta\build\Release\manta.exe .\scene\smoke_pos_size.py

REM 2. train
REM python main.py --is_3d=False --dataset=smoke_pos21_size5_f200 --res_x=96 --res_y=128 --arch=dg

REM 3. test
REM python main.py --is_train=False --load_path=MODEL_DIR --test_batch_size=100 --is_3d=False --dataset=smoke_pos21_size5_f200 --res_x=96 --res_y=128


REM ----- smoke3_vel_buo, 3D
REM ..\manta\build\Release\manta.exe .\scene\smoke3_vel_buo.py
REM python main.py --is_3d=True --dataset=smoke3_vel5_buo3_f250 --res_x=112 --res_y=64 --res_z=32 --batch_size=4 --num_worker=1 --log_step=100 --test_step=20
REM python main.py --is_train=False --load_path=MODEL_DIR --test_batch_size=5 --is_3d=True --dataset=smoke3_vel5_buo3_f250 --res_x=112 --res_y=64 --res_z=32 --batch_size=4 --num_worker=1


REM ----- smoke3_obs_buo, 3D
REM ..\manta\build\Release\manta.exe .\scene\smoke3_obs_buo.py
REM python main.py --is_3d=True --dataset=smoke3_obs11_buo4_f150 --res_x=64 --res_y=96 --res_z=64 --batch_size=3 --num_worker=1 --log_step=100 --test_step=20


REM ----- liquid_pos_size, 2D
REM ..\manta\build\Release\manta.exe .\scene\liquid_pos_size.py
REM python main.py --use_curl=False --is_3d=False --dataset=liquid_pos10_size4_f200 --res_x=128 --res_y=64


REM ----- liquid3_d_r, 3D
REM ..\manta\build\Release\manta .\scene\liquid3_d_r.py
REM python main.py --use_curl=False --is_3d=True --dataset=liquid3_d5_r10_f150 --res_x=96 --res_y=48 --res_z=96 --batch_size=3 --num_worker=1 --log_step=100 --test_step=20


REM ----- liquid3_vis, 3D
REM ..\manta\build\Release\manta ./scene/liquid3_vis.py
python main.py --use_curl=False --is_3d=True --dataset=liquid3_vis4_f150 --res_x=96 --res_y=72 --res_z=48 --batch_size=3 --num_worker=1 --log_step=100 --test_step=20


REM ----- smoke3_rot, 2D
REM ..\manta\build\Release\manta.exe .\scene\smoke3_rot.py --log_dir=data\smoke_rot_f500 --resolution_x=96 --resolution_y=128 --resolution_z=1 --open_bound=xXyY --num_dof=1


REM ----- smoke3_rot, 3D
REM ..\manta\build\Release\manta.exe .\scene\smoke3_rot.py --log_dir=data\smoke3_rot_f500 --resolution_x=48 --resolution_y=72 --resolution_z=48


REM ----- smoke3_mov, 2D
REM ..\manta\build\Release\manta.exe .\scene\smoke3_mov.py --log_dir=data\smoke_mov200_f400 --resolution_x=96 --resolution_y=128 --resolution_z=1 --open_bound=xXyY --num_dof=1
REM 1. train AE
REM python main.py --arch=ae --z_num=16 --max_epoch=10 --filter=64 --is_3d=False --dataset=smoke_mov200_f400 --res_x=96 --res_y=128

REM 2. generate latent code set
REM python main.py --is_train=False --load_path=AE_MODEL_DIR --arch=ae --z_num=16 --max_epoch=20 --is_3d=False --dataset=smoke_mov200_f400 --res_x=96 --res_y=128

REM 3. train NN
REM python main.py --arch=nn --code_path=AE_MODEL_DIR --w_size=30 --z_num=16 --filters=512 --max_epoch=200 --batch_size=1024 --is_3d=False --dataset=smoke_mov200_f400 --res_x=96 --res_y=128

REM 4. predict latent code set using trained NN
REM python main.py --is_train=False --load_path=NN_MODEL_DIR --arch=nn --code_path=AE_MODEL_DIR --w_size=30 --z_num=16 --filters=512 --max_epoch=200 --batch_size=1024 --is_3d=False --dataset=smoke_mov200_f400 --res_x=96 --res_y=128

REM 5. reconstruct velocity fields from predicted latent code set
REM python main.py --is_train=False --load_path=AE_MODEL_DIR --code_path=NN_MODEL_DIR --arch=ae --z_num=16 --max_epoch=20 --is_3d=False --dataset=smoke_mov200_f400 --res_x=96 --res_y=128


REM ----- smoke3_mov, 3D
REM ..\manta\build\Release\manta.exe .\scene\smoke3_mov.py --log_dir=data\smoke3_mov200_f400 --resolution_x=48 --resolution_y=72 --resolution_z=48
REM python main.py --arch=ae --z_num=16 --max_epoch=10 --filter=64 --is_3d=True --lr_max=0.00005 --dataset=smoke3_mov200_f400 --res_x=48 --res_y=72 --res_z=48 --batch_size=4 --num_worker=1
REM python main.py --is_train=False --load_path=AE_MODEL_DIR  --arch=ae --z_num=16 --max_epoch=20 --filter=64 --is_3d=True --dataset=smoke3_mov200_f400 --res_x=48 --res_y=72 --res_z=48 --test_batch_size=5
REM python main.py --arch=nn --code_path=AE_MODEL_DIR --w_size=30 --z_num=16 --filters=512 --max_epoch=200 --batch_size=1024 --is_3d=True --dataset=smoke3_mov200_f400 --res_x=48 --res_y=72 --res_z=48
REM python main.py --is_train=False --load_path=NN_MODEL_DIR --arch=nn --code_path=AE_MODEL_DIR --w_size=30 --z_num=16 --filters=512 --max_epoch=200 --batch_size=1024 --is_3d=True --dataset=smoke3_mov200_f400 --res_x=48 --res_y=72 --res_z=48
REM python main.py --is_train=False --load_path=AE_MODEL_DIR --code_path=NN_MODEL_DIR --arch=ae --z_num=16 --max_epoch=20 --filter=64 --is_3d=True --dataset=smoke3_mov200_f400 --res_x=48 --res_y=72 --res_z=48 --test_batch_size=5
