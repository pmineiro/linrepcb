import torch

class EasyAcc:
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.sumsq = 0

    def __iadd__(self, other):
        self.n += 1
        self.sum += other
        self.sumsq += other*other
        return self

    def __isub__(self, other):
        self.n += 1
        self.sum -= other
        self.sumsq += other*other
        return self

    def mean(self):
        return self.sum / max(self.n, 1)

    def var(self):
        from math import sqrt
        return sqrt(self.sumsq / max(self.n, 1) - self.mean()**2)

    def semean(self):
        from math import sqrt
        return self.var() / sqrt(max(self.n, 1))
    
# {'uid': '0000031909', 
#  'title': 'Girls Ballet Tutu Neon Pink\n', 
#  'content': 'High quality 3 layer ballet tutu. 12 inches in length', 
#  'target_ind': [0, 1, 192406, 1327309, 1371116, 1371888, 1461720, 1476259, 1509175, 1509181, 1509182, 1535940, 1578041, 1578155, 1604047, 1604766, 1615188, 1969579, 2030361, 2186983, 2186984, 2191027, 2227069, 2342392, 2514733, 2515122, 2515192, 2515198, 2515203, 2516838, 2516839, 2775528], 
#  'target_rel': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}

def categoryCount():
    from collections import defaultdict
    import gzip
    import json
    import zipfile
        
    counts = defaultdict(int)
    examples = 0
    
    with zipfile.ZipFile('Amazon-3M.raw.zip') as fzip:
        with fzip.open('Amazon-3M.raw/trn.json.gz') as fbin:
            with gzip.open(fbin) as f:
                for line in f:
                    obj = json.loads(line)
                    examples += 1
                    
                    for label in obj['target_ind']:
                        counts[label] += 1

    indices = { v: n for n, v in enumerate(counts) }
            
    return counts, examples, indices

def embedData():
    from sentence_transformers import SentenceTransformer
    import gzip
    import json
    import zipfile
    
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    batchsize = 20
    
    with zipfile.ZipFile('Amazon-3M.raw.zip') as fzip:
        with fzip.open('Amazon-3M.raw/trn.json.gz') as fbin:
            with gzip.open(fbin) as f:
                batchencode, batchlabels = [], []

                for line in f:
                    obj = json.loads(line)
                    batchencode.append(obj['title'])
                    batchencode.append(obj['content'])
                    batchlabels.append(obj['target_ind'])
                
                    if len(batchencode) >= batchsize:
                        embed = model.encode(batchencode)
                    
                        for n, labels in enumerate(batchlabels):
                            embtitle, embcontent = embed[2*n], embed[2*n+1]
                            yield { 'title': embtitle, 
                                    'content': embcontent, 
                                    'labels': labels }
                            batchencode, batchlabels = [], []
                                         
    if len(batchencode):
        embed = model.encode(batchencode)
                    
        for n, labels in enumerate(batchlabels):
            embtitle, embcontent = embed[2*n], embed[2*n+1]
            yield { 'title': embtitle, 
                    'content': embcontent, 
                    'labels': labels }
            batchencode, batchlabels = [], []
            
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        from tqdm.notebook import tqdm
        
        _, examples, self.indices = categoryCount()
        
        Xs = []
        ys = []
        for n, what in tqdm(enumerate(embedData()), total=examples):
            title = torch.tensor(what['title'])
            content = torch.tensor(what['content'])
            Xs.append(torch.cat((title, content)).unsqueeze(0))
            thisy = set(self.indices[label] for label in what['labels'])
            ys.append(thisy)

        self.Xs = torch.cat(Xs, dim=0)
        self.ys = ys
            
    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, index):
        ys = torch.zeros(len(self.indices)).float()
        for l in self.ys[index]:
            ys[l] = 1.0
        return self.Xs[index], ys

def loadMyDataset():
    import gzip
    
    with gzip.open(f'amazon3m.pickle.gz', 'rb') as handle:
        import pickle
        return pickle.load(handle)

class ResidualBlock(torch.nn.Module):
    def __init__(self, d, device):
        super(ResidualBlock, self).__init__()
        
        self.W = torch.nn.Parameter(torch.zeros(d, d, device=device))
        self.afunc = torch.nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
    def forward(self, X):
        return X + 0.001 * self.afunc(torch.matmul(X, self.W))

class BilinearResidual(torch.nn.Module):
    def __init__(self, dobs, daction, device, depth):
        super(BilinearResidual, self).__init__()
        
        self.block = torch.nn.Sequential(*[ResidualBlock(dobs, device) for _ in range(depth) ])
        self.W = torch.nn.Parameter(torch.zeros(dobs, daction-1, device=device))
        self.b = torch.nn.Parameter(torch.zeros(1, device=device))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, Xs, Zs):
        return torch.matmul(torch.matmul(self.block(Xs), self.W), Zs[:,:-1].T) + Zs[:,-1] + self.b
        
    def preq1(self, logits):
        return self.sigmoid(logits)

class RankOneDetset(object):
    def __init__(self, actions):
        self.actions = actions
        self.N, self.K, self.D = actions.shape
        self.device = actions.device
        
        self.batcheye = torch.eye(self.D, device=self.device).unsqueeze(0).expand(self.N, -1, -1)
        self.S = self.batcheye.clone()
        self.Sinv = self.batcheye.clone()
        self.logdetfac = torch.zeros(self.N, device=self.device)
        
    def computePhi(self, i): 
        # Sprime_a <- replace column i of S with action a where det(S)=1
        # Sprime_a = S + (a - S_i) e_i^\top = S + u v^\top
        # det(Sprime_a) = det(S) (1 + e_i^\top S^{-1} (a - S_i))
        #               = (1 - (S^{-T} e_i)^\top S_i) + (S^{-T} e_i)^\top a
        #               = 0 + \phi^\top a
        
        #Sinvtopei = torch.linalg.solve(torch.transpose(self.S, 1, 2), self.batcheye[:,:,i])
        Sinvtopei = self.Sinv[:, i, :]
        return Sinvtopei, self.logdetfac
    
    def updateCoord(self, i, fstar, astar):
        Y = torch.gather(input=self.actions, 
                         dim=1, 
                         index=astar.reshape(self.N, 1, 1).expand(self.N, 1, self.D)
                        ).squeeze(1)
        Y /= torch.exp(self.logdetfac).reshape(self.N, 1)

        # replace column i of S with y
        # -----------------------------
        # Sprime = S + (y - S_i) e_i^\top = S + u v^\top
        # Sprime^{-1} = S^{-1} - 1/(1 + v^\top S^{-1} u) (S^{-1} u) (v^\top S^{-1})^\top
        
        u = Y - self.S[:, :, i]
        Sinvu = torch.bmm(self.Sinv, u.unsqueeze(2)).squeeze(2)
        vtopSinv = self.Sinv[:, i, :]
        vtopSinvu = Sinvu[:, i].unsqueeze(1).unsqueeze(2)
        self.Sinv -= (1 / (1 + vtopSinvu)) * torch.bmm(Sinvu.unsqueeze(2), vtopSinv.unsqueeze(1))
        
        self.S[:,:,i] = Y
        thislogdet = 1/self.D * (torch.log(fstar) - self.logdetfac)
        scale = torch.exp(thislogdet).reshape(self.N, 1, 1)
        self.S /= scale
        self.Sinv *= scale
        self.logdetfac += thislogdet
    
class SpannerEG(torch.nn.Module):
    def __init__(self, actions, epsilon, tzero):
        super(SpannerEG, self).__init__()
        
        self.epsilon = epsilon
        self.tzero = tzero
        self.t = 0
        
        with torch.no_grad():
            batchactions = actions.unsqueeze(0)
            self.spanner = self._make_spanner(batchactions)
            
    def _make_spanner(self, actions):
        from math import log

        # Algorithm 4 Approximate Barycentric Identification (Awerbuch and Kleinberg, 2008)
        C = 2
        
        N, K, D = actions.shape
        device = actions.device
        #detset = NaiveDetset(actions)
        detset = RankOneDetset(actions)
        design = torch.zeros(N, D, device=device).long()
                
        for i in range(D):
            psi, _ = detset.computePhi(i)
            dets = torch.abs(torch.bmm(actions, psi.unsqueeze(2))).squeeze(2) 
            fstar, astar = torch.max(dets, dim=1)
            design[:, i] = astar
            detset.updateCoord(i, fstar, astar)
                        
        for _ in range(int(D * log(D))):
            replaced = False
            for i in range(D):
                psi, logdetfac = detset.computePhi(i)
                dets = torch.abs(torch.bmm(actions, psi.unsqueeze(2))).squeeze(2)
                fstar, astar = torch.max(dets, dim=1)
                                
                if torch.any(fstar >= C * torch.exp(logdetfac)):
                    design[:, i] = astar
                    detset.updateCoord(i, fstar, astar)
                    replaced = True
                    break
                    
            if not replaced:
                break
                
        return design

    def sample(self, fhat):
        epsilon = self.epsilon * pow(self.tzero / (self.t + self.tzero), 1/3)
        self.t += 1
        
        exploit = torch.argmax(fhat, dim=1, keepdim=True)
        exploreindex = torch.randint(low=0, high=self.spanner.shape[1], size=(fhat.shape[0], 1), device=fhat.device)
        explore = torch.gather(input=self.spanner[0,:].expand(fhat.shape[0], -1), dim=1, index=exploreindex)
        shouldexplore = (torch.rand(size=(fhat.shape[0], 1), device=fhat.device) < epsilon).long()
        sample = shouldexplore * (explore - exploit) + exploit
        return sample.squeeze(1)

class Embedding(object):
    def __init__(self, seed, naction):
        from collections import defaultdict
        
        self.seed = seed
        self.cooc = defaultdict(lambda: defaultdict(int))
        self.counts = defaultdict(int)
        self.examples = 0
        self.naction = naction
    
    def consume(self, ys):        
        self.examples += ys.shape[0]
        for row in range(ys.shape[0]):
            nonzeros = [ v.item() for v in torch.nonzero(ys[row]) ]
            for a in nonzeros:
                assert 0 <= a < self.naction, a
                self.counts[a] += 1
                for b in nonzeros:
                    assert 0 <= b < self.naction, b
                    self.cooc[a][b] += 1
            
    def fit(self, rank):
        from math import log, log1p, sqrt
        
        remap = {}
        
        # Hellinger PCA
        row_indices, col_indices, values = [], [], []
        for a, na in self.counts.items():
            if a not in remap:
                remap[a] = len(remap)
                
            for b, cooc_ab in self.cooc[a].items():
                if b not in remap:
                    remap[b] = len(remap)
                
                row_indices.append(remap[a])
                col_indices.append(remap[b])
                values.append(sqrt(cooc_ab / na))
                
        # throws "not implemented error" ... #sadlife
        #
        # coo = torch.sparse_coo_tensor([ row_indices, col_indices ], values)
        # csr = coo.to_sparse_csr()
        # return torch.svd_lowrank(csr, q=d+6, niter=2, M=None), indices

        from sklearn.decomposition import TruncatedSVD
        from scipy.sparse import coo_matrix

        coo = coo_matrix( ( values, ( row_indices, col_indices ) ), 
                          shape = (len(remap), len(remap)) )
        csr = coo.tocsr()

        svd = TruncatedSVD(n_components=rank, 
                           algorithm='randomized',
                           n_iter=2,
                           random_state=self.seed)
        svd.fit(csr)
        Z = sqrt(self.examples) * torch.tensor(svd.transform(csr))
        
        assert self.examples > 0
        defaultphat = 1 / self.examples
        bias = torch.ones(Z.shape[0]) * (log(defaultphat) - log1p(-defaultphat))
        for a, na in self.counts.items():
            assert 0 <= a < self.naction
            assert 0 < na < self.examples, (a, na, self.examples)
            phat = na / self.examples
            bias[remap[a]] = log(phat) - log1p(-phat)
            
        Z = torch.cat((Z, bias.unsqueeze(1)), dim=1)
        inverseremap = { v: k for k, v in remap.items() }
        remapTensor = torch.LongTensor([ inverseremap[n] for n in range(len(remap)) ]).unsqueeze(0)
        
        return Z, remapTensor
    
def embed(dataset, batch_size, pretrain, rank, seed):
    from tqdm.notebook import tqdm
    from math import sqrt
    import time
    torch.manual_seed(seed)
    
    splitseed = seed+1
        
    predata, _ = torch.utils.data.random_split(dataset,
                                               lengths=[ pretrain, len(dataset) - pretrain ],
                                               generator=torch.Generator().manual_seed(splitseed))
    generator = torch.utils.data.DataLoader(predata, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    embedding = Embedding(seed=seed+2, naction=len(dataset.indices))
    
    print('embed', flush=True)
    for bno, (Xs, ys) in tqdm(enumerate(generator), total=(pretrain//batch_size)):
        embedding.consume(ys)
        
    print('fit', flush=True)
    Z, remap = embedding.fit(rank)
    print('done', flush=True)
    
    return Z.float(), remap, splitseed
    
def presup(dataset, actions, initlr, tzero, batch_size, depth, pretrain, cuda, seed):
    from math import sqrt
    import time
    torch.manual_seed(seed)
    
    Zs, remap, splitseed = actions
    
    if cuda:
        Zs = Zs.cuda()
        remap = remap.cuda()
    
    predata, _ = torch.utils.data.random_split(dataset,
                                               lengths=[ pretrain, len(dataset) - pretrain ],
                                               generator=torch.Generator().manual_seed(splitseed))
    generator = torch.utils.data.DataLoader(predata, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    
    print('{:<5s}\t{:<8s}\t{:<8s}\t{:<8s}\t{:<8s}\t{:<8s}'.format('n', 'loss', 'since last', 'acc', 'acc since last', 'dt (sec)'), flush=True)
    avloss, acc, sincelast, accsincelast = EasyAcc(), EasyAcc(), EasyAcc(), EasyAcc()

    model = None
    log_loss = torch.nn.BCEWithLogitsLoss()
    
    for bno, (Xs, preys) in enumerate(generator):
        Xs, preys = Xs.to(Zs.device), preys.to(Zs.device)

        if model is None:
            model = BilinearResidual(dobs=Xs.shape[1], daction=Zs.shape[1], device=Zs.device, depth=depth)
            opt = torch.optim.Adam(( p for p in model.parameters() if p.requires_grad ), lr=initlr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = lambda t: sqrt(tzero) / sqrt(tzero + t))
            start = time.time()
            
        with torch.no_grad():
            ys = torch.gather(input=preys, dim=1, index=remap.expand(preys.shape[0], -1))        

        opt.zero_grad()
        score = model.forward(0.0001 * Xs, Zs)
        loss = log_loss(score, ys)
        loss.backward()
        opt.step()
        scheduler.step()
        
        with torch.no_grad():
            pred = torch.argmax(score, dim=1)
            ypred = torch.gather(input=ys, dim=1, index=pred.unsqueeze(1))
            acc += torch.mean(ypred).float()
            accsincelast += torch.mean(ypred).float()
            avloss += loss
            sincelast += loss

        if bno & (bno - 1) == 0:
            print('{:<5d}\t{:<8.5g}\t{:<8.5g}\t{:<8.5g}\t{:<8.5g}\t{:<8.5g}'.format(avloss.n, avloss.mean(), sincelast.mean(), 
                                                                                    acc.mean(), accsincelast.mean(), time.time() - start), 
                  flush=True)
            sincelast, accsincelast = EasyAcc(), EasyAcc()

    print('{:<5d}\t{:<8.5g}\t{:<8.5g}\t{:<8.5g}\t{:<8.5g}\t{:<8.5g}'.format(avloss.n, avloss.mean(), sincelast.mean(), 
                                                                            acc.mean(), accsincelast.mean(), time.time() - start), 
          flush=True)
    
    return model
    
def train(dataset, model, actions, initlr, tzero, epsilon, epsilontzero, batch_size, pretrain, cuda, seed):
    from math import sqrt
    import time
    torch.manual_seed(seed)
    
    Zs, remap, splitseed = actions
    
    if cuda:
        Zs = Zs.cuda()
        remap = remap.cuda()
    
    _, traindata = torch.utils.data.random_split(dataset,
                                                 lengths=[ pretrain, len(dataset) - pretrain ],
                                                 generator=torch.Generator().manual_seed(splitseed))
    generator = torch.utils.data.DataLoader(traindata, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    
    log_loss = torch.nn.BCEWithLogitsLoss()
    sampler = SpannerEG(actions=Zs, epsilon=epsilon, tzero=epsilontzero)
    opt = torch.optim.Adam(( p for p in model.parameters() if p.requires_grad ), lr=initlr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = lambda t: sqrt(tzero) / sqrt(tzero + t))
    start = time.time()
        
    print('{:<5s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}'.format(
            'n', 'loss', 'since last', 'acc', 'since last', 'reward', 'since last', 'dt (sec)'), 
          flush=True)
    avloss, sincelast, acc, accsincelast, avreward, rewardsincelast = [ EasyAcc() for _ in range(6) ]
    
    for bno, (Xs, preys) in enumerate(generator):
        Xs, preys = Xs.to(Zs.device), preys.to(Zs.device)
              
        with torch.no_grad():
            ys = torch.gather(input=preys, dim=1, index=remap.expand(preys.shape[0], -1))        

        opt.zero_grad()
        logit = model.forward(0.0001 * Xs, Zs)

        with torch.no_grad():
            sample = sampler.sample(logit)
            reward = torch.gather(input=ys, dim=1, index=sample.unsqueeze(1)).float()
            
        samplelogit = torch.gather(input=logit, index=sample.unsqueeze(1), dim=1)
        loss = log_loss(samplelogit, reward)
        loss.backward()
        opt.step()
        scheduler.step()
        
        with torch.no_grad():
            pred = torch.argmax(logit, dim=1)
            ypred = torch.gather(input=ys, dim=1, index=pred.unsqueeze(1))
            acc += torch.mean(ypred).float()
            accsincelast += torch.mean(ypred).float()
            avloss += loss
            sincelast += loss
            avreward += torch.mean(reward)
            rewardsincelast += torch.mean(reward)

        if bno & (bno - 1) == 0:
            now = time.time()
            print('{:<5d}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}'.format(
                    avloss.n, avloss.mean(), sincelast.mean(), acc.mean(), 
                    accsincelast.mean(), avreward.mean(), rewardsincelast.mean(),
                    now - start),
                  flush=True)
            sincelast, accsincelast, rewardsincelast = [ EasyAcc() for _ in range(3) ]

    now = time.time()
    print('{:<5d}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}'.format(
            avloss.n, avloss.mean(), sincelast.mean(), acc.mean(), 
            accsincelast.mean(), avreward.mean(), rewardsincelast.mean(),
            now - start),
          flush=True)

def loadEmbedding(rank):
    import gzip
    
    with gzip.open(f'amazon3m.embeds.{rank}.pickle.gz', 'rb') as handle:
        import pickle
        return pickle.load(handle)


mydata = loadMyDataset()

def doit(rank, *, cuda, initlr, tzero, banditinitlr, bandittzero, epsilon, epsilontzero, seed, prebs, banditbs):
    pretrain, depth = 50000, 2
    
    embeds = loadEmbedding(rank=rank)
    print('pretrain')
    model = presup(mydata, actions=embeds, depth=depth,
                   initlr=initlr, tzero=tzero, 
                   batch_size=prebs, pretrain=pretrain, cuda=cuda, seed=seed)
#    print('train')
#    train(mydata, model=model, actions=embeds, 
#          initlr=banditinitlr, tzero=bandittzero,
#          epsilon=epsilon, epsilontzero=epsilontzero, 
#          batch_size=banditbs, pretrain=pretrain, cuda=cuda, seed=seed)
    
def flass():
    import random

    for (initlr, tzero, 
         banditinitlr, bandittzero,
         epsilon, epsilontzero) in ( (1/160 + 1/40 * random.random(),       # initlr
                                      10 + 100 * random.random(),           # tzero
                                      1/2560 + 1/640 * random.random(),     # banditinitlr
                                      10 + 100 * random.random(),           # bandittzero
                                      1/20 + 1/5 * random.random(),         # epsilon
                                      1 + 10 * random.random(),             # epsilontzero
                                     )
                                     for _ in range(59)
                                   ):
            seed, rank, prebs, banditbs = 4545, 800, 32, 256
            print(f'doit(cuda=True, rank={rank}, seed={seed}, initlr={initlr}, tzero={tzero}, banditinitlr={banditinitlr}, bandittzero={bandittzero}, epsilon={epsilon}, epsilontzero={epsilontzero}, prebs={prebs}, banditbs={banditbs})')
            doit(cuda=True, rank=rank, seed=seed,
                 initlr=initlr, tzero=tzero,
                 banditinitlr=banditinitlr, bandittzero=bandittzero,
                 epsilon=epsilon, epsilontzero=epsilontzero,
                 prebs=prebs, banditbs=banditbs)

flass()
