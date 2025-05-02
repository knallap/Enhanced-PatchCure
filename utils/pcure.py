import torch
import time
import torch.nn.functional as F
import numpy as np

class PatchCURE(torch.nn.Module):
	def __init__(self,srf,lrf):
		super().__init__()
		self.srf = srf
		self.lrf = lrf
	def forward(self,x):
		x = self.srf(x)
		x = self.lrf(x)
		return x
	def certify(self,x,y,thres=None):
		x = self.srf(x)
		return self.lrf.certify(x,y)



class SecurePooling(torch.nn.Module):
    def __init__(self, input_size, mask_size, mask_stride, confidence_threshold=0.7, use_soft_voting=True):
        super().__init__()
        self.input_size = input_size
        self.mask_size = mask_size
        self.mask_stride = mask_stride
        self.confidence_threshold = confidence_threshold
        self.use_soft_voting = use_soft_voting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._generate_mask()

    def _generate_mask(self):
        h, w = self.input_size
        mh, mw = self.mask_size
        sh, sw = self.mask_stride

        mask_x = list(range(0, h - mh + 1, sh))
        if (h - mh) % sh != 0:
            mask_x.append(h - mh)
        mask_y = list(range(0, w - mw + 1, sw))
        if (w - mw) % sw != 0:
            mask_y.append(w - mw)

        self.masks = torch.ones(len(mask_x), len(mask_y), 1, h, w)
        for i, mx in enumerate(mask_x):
            for j, my in enumerate(mask_y):
                self.masks[i, j, :, mx:mx + mh, my:my + mw] = 0

        self.masks = self.masks.view(-1, 1, h, w).to(self.device)  # [N, 1, H, W]
        print(f"Generated {self.masks.shape[0]} masks")

    def forward(self, x):
        B, C, H, W = x.shape
        N = self.masks.shape[0]

        x_repeat = x.unsqueeze(1).repeat(1, N, 1, 1, 1).reshape(B * N, C, H, W)
        masks_repeat = self.masks.repeat(B, 1, 1, 1)  # [B*N, 1, H, W]

        masked = x_repeat * masks_repeat
        logits = masked.sum(dim=(2, 3))  # [B*N, C]
        logits = logits.view(B, N, -1)   # [B, N, C]

        if self.use_soft_voting:
            probs = torch.softmax(logits, dim=2)  # [B, N, C]
            avg_probs = probs.mean(dim=1)         # [B, C]
            return avg_probs.argmax(dim=1)
        else:
            votes = logits.argmax(dim=2)          # [B, N]
            pred = []
            for v in votes:
                values, counts = torch.unique(v, return_counts=True)
                pred.append(values[counts.argmax()])
            return torch.stack(pred)

    def certify(self, x, y):
        B, C, H, W = x.shape
        N = self.masks.shape[0]

        x_repeat = x.unsqueeze(1).repeat(1, N, 1, 1, 1).reshape(B * N, C, H, W)
        masks_repeat = self.masks.repeat(B, 1, 1, 1)  # [B*N, 1, H, W]

        masked = x_repeat * masks_repeat
        logits = masked.sum(dim=(2, 3))  # [B*N, C]
        logits = logits.view(B, N, -1)   # [B, N, C]

        predictions = logits.argmax(dim=2)  # [B, N]
        agreement = predictions.eq(y.unsqueeze(1))  # [B, N]
        return agreement.all(dim=1)  # [B]


class SecureLayer(SecurePooling): 
    # For LRF-only and LRF+SRF scenarios
    def __init__(self, lrf, input_size, mask_size, mask_stride, confidence_threshold=0.7, use_soft_voting=True):
        super().__init__(input_size, mask_size, mask_stride, confidence_threshold, use_soft_voting)
        self.lrf = lrf
    
    def _get_one_mask_logits(self, x, return_unmasked=True):
        # Unmasked logits (single full image)
        unmasked_logits = self.lrf(x).unsqueeze(-1).unsqueeze(-1) if return_unmasked else None  # [B,C,1,1]

        # Apply each mask and run through LRF
        one_masked_image = torch.einsum('bcij,ncij -> bnijc', x, self.masks.squeeze(1))  # [B, N, H, W, C]
        B, N, H, W, C = one_masked_image.shape
        one_masked_image = one_masked_image.permute(0, 1, 4, 2, 3).reshape(B * N, C, H, W)

        # LRF output on masked images
        one_mask_logits = self.lrf(one_masked_image).view(B, N, -1)  # [B, N, C]
        one_mask_logits = one_mask_logits.permute(0, 2, 1).unsqueeze(-1)  # [B, C, N, 1]

        return one_mask_logits, unmasked_logits
    
    def forward(self, x):
        unmasked_logits = self.lrf(x)
        return torch.argmax(unmasked_logits, dim=1)  # Clean prediction

    def certify(self, x, y):
        B = x.shape[0]
        N = self.masks.shape[0]

        x_repeat = x.unsqueeze(1).repeat(1, N, 1, 1, 1).reshape(B * N, x.shape[1], x.shape[2], x.shape[3])
        masks_repeat = self.masks.repeat(B, 1, 1, 1)  # [B*N, 1, H, W]

        masked = x_repeat * masks_repeat
        logits = self.lrf(masked).view(B, N, -1)  # [B, N, C]
        predictions = logits.argmax(dim=2)  # [B, N]

        agreement = predictions.eq(y.unsqueeze(1))  # [B, N]
        return agreement.all(dim=1)  # [B]


# below are some other implementation not used in main.py 
# e.g., model = PatchCURE(BagNet(),PatchGuardPooling())
class PatchGuardPooling(torch.nn.Module):
	def __init__(self,mask_size):
		super().__init__()
		self.mask_size = mask_size

	def forward(self,x):
		x = torch.clamp(x,min=0)
		unmasked_logits = torch.sum(x,dim=(2,3),keepdim=True) # [B,C,1,1]
		window_sum = F.avg_pool2d(x,kernel_size=self.mask_size, stride=(1,1), divisor_override=1)# [B,C,H',W']
		max_window_sum,_ = torch.max(window_sum.flatten(2),dim=2) # [B,C]
		masked_logits = unmasked_logits.squeeze() - max_window_sum
		#return masked_logits
		return torch.argmax(masked_logits,dim=1)
	def certify(self,x,y):
		# being lazy here, copied PatchGuard's implementation (which uses numpy and might be slow)
		x = torch.clamp(x,min=0).permute(0,2,3,1).detach().cpu().numpy()
		y = y.cpu().numpy()
		certify = torch.tensor([self._provable_masking(x[i],y[i],window_shape=self.mask_size)==2 for i in range(len(y))])
		return certify


	def _provable_masking(self,local_feature,label,clipping=-1,thres=0.,window_shape=[6,6],ds=False):
		'''
		local_feature	numpy.ndarray, feature tensor in the shape of [feature_size_x,feature_size_y,num_cls]
		label 			int, true label
		clipping 		int/float, the positive clipping value ($c_h$ in the paper). If clipping < 0, treat clipping as np.inf
		thres 			float in [0,1], detection threshold. ($T$ in the paper)
		window_shape	list [int,int], the shape of sliding window
		ds 				boolean, whether is for mask-ds

		Return 		int, provable analysis results (0: incorrect clean prediction; 1: possible attack found; 2: certified robustness )
		'''

		feature_size_x,feature_size_y,num_cls = local_feature.shape
		window_size_x,window_size_y = window_shape
		num_window_x = feature_size_x - window_size_x + 1 if not ds else feature_size_x
		num_window_y = feature_size_y - window_size_y + 1 if not ds else feature_size_y

		if clipping > 0:
			local_feature = np.clip(local_feature,0,clipping)
		else:
			local_feature = np.clip(local_feature,0,np.inf)

		global_feature = np.sum(local_feature,axis=(0,1))

		pred_list = np.argsort(global_feature,kind='stable')
		global_pred = pred_list[-1]

		if global_pred != label: # clean prediction is incorrect
			return 0

		local_feature_pred = local_feature[:,:,global_pred]

		# the sum of class evidence within each window
		in_window_sum_tensor = np.zeros([num_window_x,num_window_y,num_cls])

		for x in range(0,num_window_x):
			for y in range(0,num_window_y):
				if ds and x+window_size_x>feature_size_x:  #only happens when ds is True
					in_window_sum_tensor[x,y,:] = np.sum(local_feature[x:,y:y+window_size_y,:],axis=(0,1)) + np.sum(local_feature[:x+window_size_x-feature_size_x,y:y+window_size_y,:],axis=(0,1))
				else:
					in_window_sum_tensor[x,y,:] = np.sum(local_feature[x:x+window_size_x,y:y+window_size_y,:],axis=(0,1))


		idx = np.ones([num_cls],dtype=bool)
		idx[global_pred]=False
		for x in range(0,num_window_x):
			for y in range(0,num_window_y):

				# determine the upper bound of wrong class evidence
				global_feature_masked = global_feature - in_window_sum_tensor[x,y,:] # $t$ in the proof of Lemma 1
				global_feature_masked[idx]/=(1 - thres) # $t/(1-T)$, the upper bound of wrong class evidence 

				# determine the lower bound of true class evidence
				local_feature_pred_masked = local_feature_pred.copy()
				if ds and x+window_size_x>feature_size_x:
					local_feature_pred_masked[x:,y:y+window_size_y]=0 
					local_feature_pred_masked[:x+window_size_x-feature_size_x,y:y+window_size_y]=0 
				else:
					local_feature_pred_masked[x:x+window_size_x,y:y+window_size_y]=0 # operation $u\odot(1-w)$

				in_window_sum_pred_masked = in_window_sum_tensor[:,:,global_pred].copy()
				overlap_window_max_sum = 0
				# only need to recalculate the windows the are partially masked
				for xx in range(max(0,x - window_size_x + 1),min(x + window_size_x,num_window_x)):
					for yy in range(max(0,y - window_size_y + 1),min(y + window_size_y,num_window_y)):
						if ds and xx+window_size_x>feature_size_x:
							in_window_sum_pred_masked[xx,yy]=local_feature_pred_masked[xx:,yy:yy+window_size_y].sum()+local_feature_pred_masked[:xx+window_size_x-feature_size_x,yy:yy+window_size_y].sum()
							overlap_window_max_sum = in_window_sum_pred_masked[xx,yy] if overlap_window_max_sum<in_window_sum_pred_masked[xx,yy] else overlap_window_max_sum
						else:
							in_window_sum_pred_masked[xx,yy]=local_feature_pred_masked[xx:xx+window_size_x,yy:yy+window_size_y].sum()
							overlap_window_max_sum = in_window_sum_pred_masked[xx,yy] if overlap_window_max_sum<in_window_sum_pred_masked[xx,yy] else overlap_window_max_sum
							
				max_window_sum_pred = np.max(in_window_sum_pred_masked) # find the window with the largest sum
				if max_window_sum_pred / local_feature_pred_masked.sum() > thres: 
					global_feature_masked[global_pred]-=max_window_sum_pred
				else:
					global_feature_masked[global_pred]-=overlap_window_max_sum
					

				# determine if an attack is possible
				if np.argsort(global_feature_masked,kind='stable')[-1]!=label: 
					return 1

		return 2 #provable robustness



class CBNPooling(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self,x):
		x = torch.tanh(x*0.05-1)
		x =  torch.mean(x,dim=(2,3))
		return torch.argmax(x,dim=1)

	def certify(self,x,y):
		raise NotImplementedError()

class MRPooling(SecurePooling): # for now, for simplicty, I inherit the SecurePooling class. 
	def __init__(self,input_size,mask_size,mask_stride,thres=0):
		super().__init__(input_size,mask_size,mask_stride)
		self.C = input_size[0]*input_size[1] - mask_size[0]*mask_size[1]
		self.thres = thres

	def helper(self,x):
		one_mask_logits,_ = self._get_one_mask_logits(x)
		one_mask_logits = one_mask_logits/self.C 
		one_mask_softmax = torch.softmax(one_mask_logits,dim=1)
		one_mask_softmax = one_mask_softmax.flatten(start_dim=2)
		one_mask_conf,one_mask_pred = one_mask_softmax.max(dim=1)
		return one_mask_conf,one_mask_pred
	def forward(self,x):

		#one_mask_logits,unmasked_logits = self._get_one_mask_logits(x)
		#unmasked_pred_correct = torch.argmax(unmasked_logits.squeeze(),dim=1) == y 
		#one_mask_logits = one_mask_logits/self.C 
		#one_mask_softmax = torch.softmax(one_mask_logits,dim=1)
		#one_mask_softmax = one_mask_softmax.flatten(start_dim=2)
		#one_mask_conf,one_mask_pred = one_mask_softmax.max(dim=1)

		one_mask_conf,one_mask_pred = self.helper(x)
		#y = y.unsqueeze(-1)
		#correct = torch.logical_and(unmasked_pred_correct,~torch.any(torch.logical_and(one_mask_conf>self.thres,one_mask_pred!=y),dim=1))
		#certify = torch.logical_and(unmasked_pred_correct,torch.all(torch.logical_and(one_mask_conf>self.thres,one_mask_pred==y),dim=1))
		#correct = ~torch.any(torch.logical_and(one_mask_conf>self.thres,one_mask_pred!=y),dim=1)
		#certify = torch.all(torch.logical_and(one_mask_conf>self.thres,one_mask_pred==y),dim=1)

		pred = torch.ones_like(one_mask_conf[:,0])
		for i in range(len(pred)):
			pred_i = one_mask_pred[i][one_mask_conf[i]>self.thres]
			unique = torch.unique(pred_i)
			if len(unique) == 1:
				pred[i] = unique[0]
			else:
				pred[i] = -1
		return pred

	def certify(self,x,y):
		one_mask_conf,one_mask_pred = self.helper(x)
		y = y.unsqueeze(-1)
		certify = torch.all(torch.logical_and(one_mask_conf>self.thres,one_mask_pred==y),dim=1)
		return certify
class MRPC(MRPooling): # Minority Reports + PatchCURE
	def __init__(self,lrf,input_size,mask_size,mask_stride,thres=0):
		super().__init__(input_size,mask_size,mask_stride,thres)
		self.lrf = lrf #TODO, we could unify the API for SecureLayer and SecurePooling
	def _get_one_mask_logits(self,x,return_unmasked=True):
		unmasked_logits = self.lrf(x).unsqueeze(-1).unsqueeze(-1) if return_unmasked else None # [B,C,1,1] 
		one_masked_image = torch.einsum('bcij,nmcij -> bnmcij',x,self.mask_list) #[B,N1,N2,C,W,H]
		B,N1,N2,C,W,H = one_masked_image.shape
		one_masked_image = one_masked_image.flatten(end_dim=2) #[B*N1*N2,C,W,H]
		one_mask_logits = self.lrf(one_masked_image).view([B,N1,N2,-1]).permute(0,3,1,2)
		return one_mask_logits,unmasked_logits

