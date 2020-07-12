# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 18:05:48 2020

@author: caster
"""

import VAERecommender_search as se
import VAERecommender_VAE as vae
import VAERecommender_prepare_data as ds
import torch
import numpy as np
import copy
import re
import time

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('CUDA is available,',torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HybridRecommender:
    
    def __init__(self, data, weights_path = './CVAE_weights_VAE.pth', retrain = False, experiment = False):
        self.tagSearcher = se.TagSearcher(data.vndbTagDict, data.tagTree, data.releases)
        self.CVAE = vae.CVAE(data.dataset.countOfVNs).to(device)
        if not retrain:
            vae.loadModel(self.CVAE, weights_path)
            if experiment:
                vae.test(self.CVAE, data.dataset)
        else:
            vae.train(self.CVAE, data.dataset, experiment = experiment)
            vae.saveModel(self.CVAE, weights_path)
        self.maxId = data.dataset.maxId if data.dataset.maxId > self.tagSearcher.maxId else self.tagSearcher.maxId

    def _getFrontendDataRepresentation(self, topId, topId_score, data, fullInfo):
        if len(topId) != 0:
            if fullInfo:
                start = time.time()
                res = []
                regexp = r'\[url=(.*?)\](.*?)\[/url\]'
                replacement = r'<a href=\1>\2</a>'
                for vnId in topId:
                    vn = copy.deepcopy(data.frontendData.getVNDataByID(vnId))
                    vn.length = data.frontendData.lenTextRepr[vn.length - 1]
                    vn.desc = vn.desc.replace('\\n',' ')
                    vn.desc = re.sub(regexp, replacement, vn.desc)
                    languages = []
                    for language in vn.languages:
                        languages.append(data.frontendData.languageTextRepr[language])
                    vn.languages = ' '.join(languages)
                    res.append(vn)
                print('full:', time.time() - start)
                return res, topId_score
            else:
                return topId, topId_score
        else:
            return [],[]
            
        
    def recommendKItems(self, weightOfCVAE, userId, dislikes, likes, maxSpoilerLevel, isAdult, requestedLanguage, data, k = 10, fullInfo = False):
        res = np.zeros((self.maxId+1,))
        start = time.time()
        if not isAdult:
            dislikes.append(23)
        relevantVNIds = self.tagSearcher.search(res, (1 - weightOfCVAE), dislikes, likes, maxSpoilerLevel = maxSpoilerLevel)
        print('ts:',time.time() - start)
        start = time.time()
        if(userId != -1):
            vae.getVAERatings(self.CVAE, res, weightOfCVAE, userId, data.dataset)
        print('vae:',time.time() - start)
        '''
        if not isAdult:
            data.dataset.removeNSFWContentFromRecommendation(res)
        '''
        self.tagSearcher.removeUnrelevantTagsByTagsAndAgeRatingAndLanguages(res, relevantVNIds, data.frontendData.allVNIds, isAdult, requestedLanguage)
        start = time.time()
        res_ids = np.argsort(res)[::-1]
        res_score = np.sort(res)[::-1]
        print('rest:',time.time() - start)
        topId, topId_score = res_ids[:k], res_score[:k]
        topId, topId_score = topId[topId_score > 0], topId_score[topId_score > 0]
        return self._getFrontendDataRepresentation(topId, topId_score, data, fullInfo)
    
        
        
class RecommenderData:
    
    def __init__(self, pathVN = 'vn', pathTags_vn = 'tags_vn',\
                 pathTags_parents = 'tags_parents', pathVotes = 'vndb-votes-2020-01-06',\
                 pathUsers = 'users', pathTags = 'tags',\
                 pathReleases = 'releases', pathReleasesLang = 'releases_lang', pathReleasesVN = 'releases_vn',\
                 pathLanguages = 'languages'):

        self.releases = ds.ReleaseProcessor(pathReleases, pathReleasesLang, pathReleasesVN, pathLanguages)
        self.dataset = ds.RatingsDataset(pathVotes, pathVN)
        self.vndbTagDict = ds.VNTagDict(pathTags_vn, pathTags)
        self.tagTree = ds.TagTree(pathTags_parents)
        self.frontendData = ds.VNDBLib(self.dataset.vnIds.tolist(), list(self.vndbTagDict.keys()), self.releases, pathLanguages, pathVN)        #!!!!!!!!!!!!!!!!!
        ds.prepareClientSearchJSON(pathUsers, pathTags, './static/jsonNotJSON', self.dataset.votes.keys(), self.vndbTagDict.allAvailableTags, self.tagTree)
        

if __name__ == '__main__':
    
    data = RecommenderData(pathVN = './dump/db/vn', pathTags_vn = './dump/db/tags_vn',\
                                 pathTags_parents = './dump/db/tags_parents', pathTags= './dump/db/tags',
                                 pathVotes = './dump/db/votes', pathUsers='./dump/db/users',
                                 pathReleases= './dump/db/releases', pathReleasesLang='./dump/db/releases_lang',\
                                pathReleasesVN= './dump/db/releases_vn', pathLanguages='./dump/languages')      
    print('data ready')
    rec = HybridRecommender(data, experiment = True, retrain = False)
    print('model ready')
    
    '''
    start = time.time()
#167963
#127198
#151834
#110743
    res, score = rec.recommendKItems(0.5, 167963, [23], {147: 2.}, 0., data)
    for i in range(len(res)):
        print(res[i], score[i])
    print('time:', time.time() - start)
    '''