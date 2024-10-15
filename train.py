# Training a NestFuse network
# auto-encoder

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from model_net import model_net
from args_fusion import args
import pytorch_msssim


EPSILON = 1e-5


def main():
	original_imgs_path = utils.list_images(args.dataset_ir)
	train_num = 80000
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)
	img_flag = args.img_flag
	alpha_list = [1]
	w_all_list = [[6.0, 3.0]]

	for w_w in w_all_list:
		w1, w2 = w_w
		for alpha in alpha_list:
			train(original_imgs_path, img_flag, alpha, w1, w2)


def train(original_imgs_path, img_flag, alpha, w1, w2):

	batch_size = args.batch_size
	# load network model
	nc = 1
	input_nc = args.input_nc
	output_nc = args.output_nc
	LiMFusion_model = model_net(input_nc, output_nc)

	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		LiMFusion_model.load_state_dict(torch.load(args.resume))
		
	print(LiMFusion_model)
	optimizer = Adam(LiMFusion_model.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()

	if args.cuda:
		LiMFusion_model.cuda()

	tbar = trange(args.epochs)
	print('Start training.....')

	# creating save path
	temp_path_model = os.path.join(args.save_model_dir)
	temp_path_loss  = os.path.join(args.save_loss_dir)
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	temp_path_model_w = os.path.join(args.save_model_dir, str(w1))
	temp_path_loss_w  = os.path.join(args.save_loss_dir, str(w1))
	if os.path.exists(temp_path_model_w) is False:
		os.mkdir(temp_path_model_w)

	if os.path.exists(temp_path_loss_w) is False:
		os.mkdir(temp_path_loss_w)

	L_spa = utils.L_spa()
	Loss_feature = []
	Loss_grad = []
	Loss_all = []
	count_loss = 0
	all_grad_loss = 0.
	all_fea_loss = 0.
	all_spatial_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
		LiMFusion_model.train()
		LiMFusion_model.cuda()
        
        
		count = 0
		for batch in range(batches):
			image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img_ir = utils.get_train_images_auto(image_paths_ir, height=args.HEIGHT, width=args.WIDTH, mode=img_flag)

			image_paths_vi = [x.replace('IR', 'VI_RGB') for x in image_paths_ir]
			img_vi = utils.get_train_images_auto(image_paths_vi, height=args.HEIGHT, width=args.WIDTH, mode=img_flag)

			image_paths_ir_H = [x.replace(r"D:\FPDE\MSRS\IR", r'D:\FPDE\output\IR_H') for x in image_paths_ir]
			ir_H = utils.get_train_images_auto(image_paths_ir_H, height=args.HEIGHT, width=args.WIDTH, mode=img_flag)

			#print(image_paths_ir_H)

			image_paths_ir_L = [x.replace(r"D:\FPDE\MSRS\IR", r'D:\FPDE\output\IR_L') for x in image_paths_ir]
			ir_L = utils.get_train_images_auto(image_paths_ir_L, height=args.HEIGHT, width=args.WIDTH, mode=img_flag)

			image_paths_ir_hog = [x.replace(r"D:\FPDE\MSRS\IR", r'D:\FPDE\output\IR_hog') for x in image_paths_ir]
			ir_map = utils.get_train_images_auto(image_paths_ir_hog, height=args.HEIGHT, width=args.WIDTH, mode=img_flag)

			image_paths_vi_L = [x.replace(r"D:\FPDE\MSRS\IR", r'D:\FPDE\output\VIS_L') for x in image_paths_ir]
			vi_L = utils.get_train_images_auto(image_paths_vi_L, height=args.HEIGHT, width=args.WIDTH, mode=img_flag)

			image_paths_vi_H = [x.replace(r"D:\FPDE\MSRS\IR", r'D:\FPDE\output\VIS_H') for x in image_paths_ir]
			vi_H = utils.get_train_images_auto(image_paths_vi_H, height=args.HEIGHT, width=args.WIDTH, mode=img_flag)

			image_paths_vi_hog = [x.replace(r"D:\FPDE\MSRS\IR", r'D:\FPDE\output\VIS_hog') for x in image_paths_ir]
			vi_map = utils.get_train_images_auto(image_paths_vi_hog, height=args.HEIGHT, width=args.WIDTH, mode=img_flag)

			count += 1
			optimizer.zero_grad()
			img_ir = Variable(img_ir, requires_grad=False)
			img_vi = Variable(img_vi, requires_grad=False)

			ir_H = Variable(ir_H, requires_grad=False)
			ir_L = Variable(ir_L, requires_grad=False)
			ir_map = Variable(ir_map, requires_grad=False)
			vi_L = Variable(vi_L, requires_grad=False)
			vi_H = Variable(vi_H, requires_grad=False)
			vi_map = Variable(vi_map, requires_grad=False)

			if args.cuda:
				img_ir = img_ir.cuda()
				img_vi = img_vi.cuda()
				ir_H = ir_H.cuda()
				ir_L = ir_L.cuda()
				ir_map = ir_map.cuda()
				vi_L = vi_L.cuda()
				vi_H = vi_H.cuda()
				vi_map = vi_map.cuda()
			# encoder
			# ir_L = utils.fpde(img_ir)
			# vi_L = utils.fpde(img_vi)
			# ir_H = img_ir - ir_L
			# vi_H = img_vi - vi_L
			#
			# ir_map = utils.get_hop_weight_map2(ir_H)
			# vi_map = utils.get_hop_weight_map2(vi_H)

			f_H = LiMFusion_model.encoder_H(ir_H, vi_H, img_ir, img_vi)[0]
			f_L = LiMFusion_model.encoder_L(ir_L, vi_L, ir_map, vi_map)[0]

			# decode
			outputs = [f_H + f_L]

			x_ir = Variable(img_ir.data.clone(), requires_grad=False)
			x_vi = Variable(img_vi.data.clone(), requires_grad=False)

			x_ir_L = Variable(ir_L.data.clone(), requires_grad=False)
			x_vi_L = Variable(vi_L.data.clone(), requires_grad=False)
			x_ir_H = Variable(ir_H.data.clone(), requires_grad=False)
			x_vi_H = Variable(vi_H.data.clone(), requires_grad=False)

			######################### LOSS FUNCTION #########################
			loss1_value = 0.
			loss2_value = 0.
			loss3_value = 0.
			for output in outputs:

				grad_loss_temp = utils.gradient_loss(output, x_ir, x_vi)
				pixel_loss_temp = mse_loss(f_L, torch.max(x_ir_L, x_vi_L))

				spatial_loss_temp = torch.mean(L_spa(output, x_ir)) + torch.mean(L_spa(output, x_vi))


				loss1_value += alpha * (grad_loss_temp) * 10
				loss2_value += pixel_loss_temp * 1
				loss3_value += spatial_loss_temp * 0.1

			loss1_value /= len(outputs)
			loss2_value /= len(outputs)
			loss3_value /= len(outputs)

			# total loss
			total_loss = loss1_value + loss2_value + loss3_value
			total_loss.backward()
			optimizer.step()

			all_fea_loss += loss2_value.item() # 
			all_grad_loss += loss1_value.item() #
			all_spatial_loss += loss3_value.item()  #
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\t Alpha: {} \tW-IR: {}\tEpoch {}:\t[{}/{}]\t ssim loss: {:.6f}\t fea loss: {:.6f}\t sa loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), alpha, w1, e + 1, count, batches,
								  all_grad_loss / args.log_interval,
								  all_fea_loss / args.log_interval,
								  all_spatial_loss / args.log_interval,
								  (all_fea_loss + all_grad_loss + all_spatial_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_grad.append( all_grad_loss / args.log_interval)
				Loss_feature.append(all_fea_loss / args.log_interval)
				Loss_all.append((all_fea_loss + all_grad_loss) / args.log_interval)
				count_loss = count_loss + 1
				all_grad_loss = 0.
				all_fea_loss = 0.
				all_spatial_loss = 0.

			if (batch + 1) % (200 * args.log_interval) == 0:
				# save model
				LiMFusion_model.eval()
				LiMFusion_model.cpu()
                
				save_model_filename = "Epoch_" + str(e) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".model"
				save_model_path = os.path.join(args.save_model_dir, save_model_filename)
				torch.save(LiMFusion_model.state_dict(), save_model_path)
                
				# save loss data
				# pixel loss
				loss_data_grad = Loss_grad
				loss_filename_path = temp_path_loss_w + "/loss_ssim_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				scio.savemat(loss_filename_path, {'loss_ssim': loss_data_grad})
				# SSIM loss
				loss_data_fea = Loss_feature
				loss_filename_path = temp_path_loss_w + "/loss_fea_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				scio.savemat(loss_filename_path, {'loss_fea': loss_data_fea})
				# all loss
				loss_data = Loss_all
				loss_filename_path = temp_path_loss_w + "/loss_all_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				scio.savemat(loss_filename_path, {'loss_all': loss_data})

				LiMFusion_model.train()
				LiMFusion_model.cuda()
                				
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

		# ssim loss
		loss_data_grad = Loss_grad
		loss_filename_path = temp_path_loss_w + "/Final_loss_ssim_epoch_" + str(
			args.epochs) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
		scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_grad})
		loss_data_fea = Loss_feature
		loss_filename_path = temp_path_loss_w + "/Final_loss_2_epoch_" + str(
			args.epochs) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
		scio.savemat(loss_filename_path, {'final_loss_fea': loss_data_fea})
		# SSIM loss
		loss_data = Loss_all
		loss_filename_path = temp_path_loss_w + "/Final_loss_all_epoch_" + str(
			args.epochs) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
		scio.savemat(loss_filename_path, {'final_loss_all': loss_data})
		# save model
		LiMFusion_model.eval()
		LiMFusion_model.cpu()
        
		save_model_filename = "Final_epoch_" + str(args.epochs) + "_alpha_" + str(alpha) + "_wir_" + str(
			w1) + "_wvi_" + str(w2) + ".model"
		save_model_path = os.path.join(args.save_model_dir, save_model_filename)
		torch.save(LiMFusion_model.state_dict(), save_model_path)
        
		print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
	try:
		if not os.path.exists(args.vgg_model_dir):
			os.makedirs(args.vgg_model_dir)
		if not os.path.exists(args.save_model_dir):
			os.makedirs(args.save_model_dir)
	except OSError as e:
		print(e)
		sys.exit(1)



if __name__ == "__main__":
	main()
