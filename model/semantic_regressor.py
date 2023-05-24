import numpy as np
import os
from load.dataset import Dataset

from load.data_loaders.karate import KaratePoses
from utils.parser_util import classify_args, train_args
from utils.fixseed import fixseed
from load.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import torch


class SemanticRegressor(nn.Module):
    def __init__(self, input_dim=512, output_dim=5):
        super(SemanticRegressor, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):

        #print(x)

        x = self.linear(x)

        #print(x)
        #sm = nn.Softmax(dim=-1)
        #x = sm(x)
        return x


def load_dataset(args, max_frames, n_frames):
    ds_loader = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              test_participant='b0372',
                              split='test')
    ds_loader.fixed_length = n_frames
    return ds_loader


if __name__ == "__main__":

    kp = KaratePoses()
    kp.num_frames = 100


    pos = kp[0]

    args = classify_args()

    fixseed(args.seed)
    max_frames = 100
    fps = 25
    n_frames = max_frames   # min(max_frames, int(args.motion_length * fps))

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)

    print(len(data.dataset))

    #iterator = iter(data)

    #data_batch, model_kwargs = next(iterator)

    #total_num_samples = args.num_samples * args.num_repetitions

    #print(data_batch[0].shape, model_kwargs['y']['labels'][0])

    # TODO: get original_motion encode it with the semantic encoder and then train classifier with the extracted labels

    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    #if args.guidance_param != 1:
    #    model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.requires_grad_(False)
    model.eval()  # disable random masking

    regressor = SemanticRegressor()
    regressor.to(dist_util.dev())
    #criterion =  #torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(regressor.parameters(), lr=0.01)

    all_loss = []
    all_acc = []
    for epoch in range(100): # 300
        total_correct = 0
        total_instances = 0
        epoch_loss = []
        for motion, cond in tqdm(data):
            cond['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in
                         cond['y'].items()}

            og_motion = cond['y']['original_motion']
            target = cond['y']['labels'].squeeze() #.unsqueeze(dim=1)

            #print(target)
            #exit()

            #print(target)
            #exit()

            #og_motion = og_motion.to(dist_util.dev())

            #print(og_motion[0])
            #print(og_motion.shape)

            #exit()

            with torch.no_grad():
                #print(og_motion)
                # TODO: problem: for some reason the semantic encoder output the same values for all inputs in a batch
                # Fixed by not having an own class for the semantic encoder but do it inside the mdm class.
                # Strange that this makes a difference....
                #emb = model.semantic_encoder(og_motion)
                emb = model.encode_semantic(og_motion)
                #print(emb)
                #print(emb.shape)
            #exit()

            output = regressor(emb)

            classifications = torch.argmax(output, dim=-1)
            labels_idxs = torch.argmax(target, dim=-1)

            #print(classifications)
            #print(labels_idxs)


            correct_predictions = sum(classifications==labels_idxs).item()

            #print(classifications==labels_idxs)
            #print(correct_predictions)
            #exit()
            total_correct += correct_predictions
            #print(len(og_motion))
            #exit()
            total_instances += len(og_motion) # not shape?

            #if epoch == 0:
                #print(output)
                #print(target)
                #print('---')
            #    print(correct_predictions)
                #print(len(og_motion))
                #exit()

            #print(output)
            #print(target)
            #exit()

            #print(output, target)
            #loss = F.binary_cross_entropy(output, target)
            loss = F.binary_cross_entropy_with_logits(output, target)
            epoch_loss.append(loss.item())
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        print(np.mean(epoch_loss))
        all_loss.append(np.mean(epoch_loss))

        #exit()

        all_acc.append(total_correct/total_instances)
        print(total_correct/total_instances)

    print(all_loss)


    print('works till here')


    # if is_using_data:
    #iterator = iter(data)



    #device = torch.device("cpu")
    #if torch.cuda.is_available() and dist_util.dev() != 'cpu':
    #    device = torch.device(dist_util.dev())
    #print(f'device: {device}')
    #data_batch = data_batch.to(device)
    #model_kwargs['y'] = {key: val.to(device) if torch.is_tensor(val) else val for key, val in model_kwargs['y'].items()}

