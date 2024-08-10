import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler 
from sksurv.metrics import concordance_index_censored
from torchmetrics.wrappers import BootStrapper
from torchmetrics import Metric
from tqdm import tqdm

def bootstrap_survival(model, dataset, n_iterations=10, batch_size = None, credible_interval=0.95, device = None):
    results = []
    model.eval()
    model = model.to(device)
    num_samples = len(dataset) 
    print(' *** bootstrapping survival analysis...dataset size: ', len(dataset))
    for _ in tqdm(range((n_iterations))):
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

        all_risk_scores = np.zeros((len(dataloader)))
        all_censorships = np.zeros((len(dataloader)))
        all_event_times = np.zeros((len(dataloader)))

        for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            if torch.cuda.is_available():
                # print('cuda is available')
                data_WSI = data_WSI.cuda()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            with torch.no_grad():
                hazards, S = model(data_WSI)
            # results
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        results.append(c_index)
    mean_c_index = np.mean(results)
    ci_lower = np.percentile(results, (1 - credible_interval) / 2 * 100)
    ci_upper = np.percentile(results, (1 + credible_interval) / 2 * 100)

    return mean_c_index, ci_lower, ci_upper



class ConcordanceIndexCensored(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("censorships", default=[], dist_reduce_fx=None)
        self.add_state("event_times", default=[], dist_reduce_fx=None)
        self.add_state("risk_scores", default=[], dist_reduce_fx=None)

    def update(self, censorships, event_times, risk_scores):
        # Concatenate new predictions, targets, and event indicators to the existing lists
        self.censorships = censorships
        self.event_times = event_times
        self.risk_scores = risk_scores

    def compute(self):
        c_index = concordance_index_censored(self.censorships, self.event_times, self.risk_scores, tied_tol=1e-08)[0]
        c_index = torch.tensor(c_index, dtype=torch.float32)
        self.reset()
        return c_index

    def reset(self):
        # Reset the state by clearing the lists
        self.preds = []
        self.targets = []
        self.event_indicator = []



def bootstrap_survivalv2(model, dataset, n_iterations=10, device = None):
    model.eval()
    model = model.to(device)
    print(' *** bootstrapping survival analysis...dataset size: ', len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
    all_risk_scores = np.zeros((len(dataloader)))
    all_censorships = np.zeros((len(dataloader)))
    all_event_times = np.zeros((len(dataloader)))

    for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
        if torch.cuda.is_available():
            # print('cuda is available')
            data_WSI = data_WSI.cuda()
            data_Label = data_Label.type(torch.LongTensor).cuda()
            data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
        with torch.no_grad():
            hazards, S = model(data_WSI)
        # results
        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = data_Censorship.item()
        all_event_times[batch_idx] = data_Event

    all_risk_scores = torch.from_numpy(all_risk_scores)
    all_censorships = torch.from_numpy(all_censorships)
    all_event_times = torch.from_numpy(all_event_times)
    
    C_index = ConcordanceIndexCensored()
    bootstrapper = BootStrapper(
    base_metric=C_index,
    num_bootstraps=n_iterations,  
    mean=True,           
    std=True,             
    sampling_strategy="multinomial"
    )   
    
    bootstrapper.update((1 - all_censorships).to(torch.bool), all_event_times, all_risk_scores)
    scores = bootstrapper.compute()
    mean_c_index = scores['mean']
    std_c_index = scores["std"]

    ci_lower = mean_c_index - 1.96 * std_c_index
    ci_upper = mean_c_index + 1.96 * std_c_index
    
    return mean_c_index, ci_lower, ci_upper
