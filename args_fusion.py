
class args():

	# training args
	epochs = 60 #"number of training epochs, default is 2"
	batch_size =16#8 #"batch size for training, default is 4"
	dataset_ir = r"D:\FPDE\MSRS\IR"
	dataset_vi = r"D:\FPDE\MSRS\VI_RGB"
	dataset_ir_H = r"D:\FPDE\output\IR_H"
	HEIGHT = 256
	WIDTH = 256

	input_nc = 1
	output_nc = 1
	img_flag = 'L'#'RGB'#

	save_model_dir = "models" #"path to folder where trained model will be saved."
	save_loss_dir = "models/loss"  # "path to folder where trained model will be saved."

	image_size = 256 #"size of training images, default is 256 X 256"
	cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"
	ssim_weight = [1,10,100,1000,10000]
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

	lr = 1e-4 #"learning rate, default is 0.0001"
	lr_light = 1e-4  # "learning rate, default is 0.0001"
	log_interval = 5 #"number of images after which the training loss is logged, default is 500"
	resume = None#"./models/Final_epoch_60_alpha_1_wir_6.0_wvi_3.0.model" #"./models/BTSFusion.model"
	resume_auto_en = None
	resume_auto_de = None
	resume_auto_fn = None

	model_path_gray = "./models/Final_epoch_60_alpha_1_wir_6.0_wvi_3.0.model"
	model_path_gray_test = "./models/Final_epoch_60_alpha_1_wir_6.0_wvi_3.0.model"




