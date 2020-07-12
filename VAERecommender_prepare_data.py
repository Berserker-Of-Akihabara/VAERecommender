# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 23:40:17 2020

@author: caster
"""

import torch
import numpy as np
import torch.utils.data
import json
import csv
import os

''''
class ScoredVN:
    
    def __init__(self, vnId, score):
        self.id = vnId
        self.score = score
        
    def __str__(self):
        return str(self.id)+' '+str(self.score)
'''


class TagStat(dict):
    '''Class to store general statistics for tag of given VN.\n
    Args:
        votes (list of TagVote) - votes for tags in one VN.
    '''
    def __init__(self, votes):
        super(TagStat, self).__init__()
        self['r'] = np.mean([vote.relevance for vote in votes])
        sp = np.mean([vote.spoiler for vote in votes])
        self['s'] = sp if not np.isnan(sp) else 0

        
class TagVote:
    '''Class of one user's single vote of tag.\n
    Args:
        lineParts (list) - line of vote
        defaultSpoilerDict (dict) - default spoiler values, which used when user don't specified it
    '''
    
    def __init__(self, lineParts, defaultSpoilerDict):
        self.tagId = int(lineParts[0])
        self.vnId = int(lineParts[1])
        self.relevance = float(lineParts[3])
        self.spoiler = float(lineParts[4]) if lineParts[4]!='\\N' else defaultSpoilerDict[self.tagId]
        
class VNTagDict(dict):
    '''Dict, that stores TagStat for every tag of every VN. Basicaly, dict of dicts.\n
        Args:
            pathTagsVN - path to tags_vn
            pathTags - path to tags'''
    
    def __init__(self, pathTagsVN = 'tags_vn', pathTags = 'tags'):
        super(VNTagDict, self).__init__()
        defaultSpoilerDict = self._getTagDefaulSpoilerDictFromFile(pathTags)
        self.allAvailableTags = self._readTagRatingsFromFile(pathTagsVN, defaultSpoilerDict)
    
    def _readTagRatingsFromFile(self, path, defaultSpoilerDict):
        '''Fills VNTagDict.\n
        Args:
            path (str) - path to tags_vn
            defaultSpoilerDict (dict) - default spoiler values, which used when user don't specified it
        Returns:
            allAvailableTags (set) - all available ids of tags
        '''
        allAvailableTags = set()
        file = open(path, 'r')
        lines = file.readlines()
        for line in lines:
            if line[-2] != 't':                     #ignore "ignored" data from dump
                lineParts = line.split('	')
                currVote = TagVote(lineParts, defaultSpoilerDict)
                if currVote.vnId not in self.keys():
                    self[currVote.vnId] = dict()
                if currVote.tagId not in self[currVote.vnId]:
                    self[currVote.vnId][currVote.tagId] = list()
                self[currVote.vnId][currVote.tagId].append(currVote)
        for vn in self.keys():
            tagsOfVN = list(self[vn].keys())      #make list of tags of current VN
            for tag in tagsOfVN:
                allAvailableTags.add(tag)
                currTagStat = TagStat(self[vn][tag])
                if currTagStat['r'] > 0:
                    self[vn][tag] = currTagStat
                else:
                    self[vn].pop(tag, 0)
        file.close()
        return allAvailableTags
    
    def _getTagDefaulSpoilerDictFromFile(self, path):
        '''Method to get default spoiler dict. Values from this dict is used when user not specified spoiler level.\n
        Args:
            path (str) - path to tags
        Returns:
            defaultSpoilerDict (dict) - tagId:defaultSpoilerValue pairs
        '''
        defaultSpoilerDict = dict()
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                lineParts = line.split('	')
                defaultSpoilerDict[int(lineParts[0])] = float(lineParts[5])
        return defaultSpoilerDict

    def writeTagDataToJSON(self):
        tagFile = open('tag_data.json', 'w')
        json.dump(self, tagFile)
        tagFile.close()

class ReleaseProcessor:

    def __init__(self, pathReleases, pathReleasesLang, pathReleasesVN, pathLanguages):
        releaseToVNDict = self._readReleaseToVN(pathReleasesVN)
        self.dictOfVNsByLanguage = self._readLangData(pathReleasesLang, pathLanguages, releaseToVNDict)
        self.nsfwVNs = self._getVNsWithNSFW(pathReleases, releaseToVNDict)

    def _readReleaseToVN(self, path):
        res = dict()
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.split('	')
                release_Id = int(line[0])
                vnId = int(line[1])
                if release_Id not in res:
                    res[release_Id] = list()
                res[release_Id].append(vnId)
        return res

    def _readLangData(self, path, pathLanguages, releaseToVNDict):
        dictOfVNsByLanguage = dict()
        languagesShortToIdx = dict()
        with open(pathLanguages, 'r') as file:
            lines = file.readlines()
            i = 0
            for line in lines:
                line = line.split(' ')
                languagesShortToIdx[line[0] + '\n'] = i
                i += 1
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.split('	')
                short_repr_lang = line[1]
                language = languagesShortToIdx.get(short_repr_lang, -1)
                if language not in dictOfVNsByLanguage:
                    dictOfVNsByLanguage[language] = list()
                release_Id = int(line[0])
                vnId_in_release = releaseToVNDict[release_Id]
                for vnId in vnId_in_release:
                    if vnId not in dictOfVNsByLanguage[language]:
                        dictOfVNsByLanguage[language].append(vnId)
        for language in dictOfVNsByLanguage.keys():
            dictOfVNsByLanguage[language] = np.asarray(dictOfVNsByLanguage[language])
        return dictOfVNsByLanguage

    def _getVNsWithNSFW(self, path, releaseToVNDict):
        res = set()
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                line = line.split('	')
                if line[9] == '18':
                    release_Id = int(line[0])
                    vnId_in_release = releaseToVNDict[release_Id]
                    for vnId in vnId_in_release:
                        res.add(vnId)
        return np.array(list(res))
    
    def getLangTextRepresentation(self, pathLanguages):
        languagesIdxToLong = dict()
        with open(pathLanguages, 'r') as file:
            lines = file.readlines()
            i = 0
            for line in lines:
                line = line.split(' ')
                languagesIdxToLong[i] = line[1][:-1]
                i += 1
            languagesIdxToLong[-1] = 'Other'
            languagesIdxToLong[-2] = 'Any'
        return languagesIdxToLong

class VN:
    '''Class to store general VN data of single VN. All frontend-related data contained here.\n
    Args:
        line (str) - line of data from dump(vn file)
    '''
    
    def __init__(self, line, releases):
        line = line.split('	')
        self.id = int(line[0])
        self.title = line[1]
        self.original = line[2]
        self.alias = line[3]
        self.length = int(line[4])
        self.img_nsfw = True if line[5] == "t" else False
        self.image = line[6][4:-1]
        self.desc = line[7]
        self.languages = list()
        for language in releases.dictOfVNsByLanguage:
            if self.id in releases.dictOfVNsByLanguage[language]:
                self.languages.append(language)
        
    def __str__(self):
        return "{0}\n{1}\n{2}\n{3}".format(self.id,self.title,self.img_nsfw,self.desc)
    
    '''
    def _setTags(self, tagDict):
        self.tags = tagDict
    '''
        
'''
class VNDBTagLib:           #Only useful method moved to VNTagDict
    
    def __init__(self, pathTagsVN = 'tags_vn', pathTags = 'tags'):
        self.tagDict = VNTagDict(pathTagsVN, pathTags)
        self.writeTagDataToJSON()
        
    def writeTagDataToJSON(self):
        tagFile = open('tag_data.json', 'w')
        json.dump(self.tagDict, tagFile)
        tagFile.close()
'''
    
    
class VNDBLib:
    """Class to contain general data of all VNs.\n
    Args:
        vnIdsVAE (list) - unique vn ids for VAE
        vnIdsSearch (list) - unique vn ids for Search
        releases (dict) - releases data
        pathLanguages(str) - path to languages
        path (str) - path to vn
    """
    
    def __init__(self, vnIdsVAE, vnIdsSearch, releases, pathLanguages = 'languages', path = 'vn'):
        self.allVNIds = np.array(list(set(vnIdsVAE + vnIdsSearch)))
        self.data = dict.fromkeys(self.allVNIds)
        print(len(vnIdsVAE))
        print(len(vnIdsSearch))
        print(len(self.data.keys()))
        self._getDataFromFile(path, releases)
        self.lenTextRepr = ['Very short (< 2 hours)', 'Short (2 - 10 hours)', 'Medium (10 - 30 hours)', 'Long (30 - 50 hours)', 'Very long (> 50 hours)']
        self.languageTextRepr = releases.getLangTextRepresentation(pathLanguages)
    
    def _getDataFromFile(self, path, releases):
        file = open(path, 'r', encoding = "utf-8", newline = '\n')
        lines = file.readlines()
        for line in lines:
            currVN = VN(line, releases)
            if currVN.id in self.data.keys():
                self.data[currVN.id] = currVN
        file.close()
        
        
    def getVNDataByID(self, vnId):
        return self.data[vnId]

class Vote:
    '''
    Contains single vote of single user for VN rating.\n
    Args:
        line (str) - string representation from votes file.
        delimiter (str) - delimiter of features in file
        vnIds (ndarray) - array of all unique vnIds
    '''
    
    def __init__(self, line, delimiter, vnIds):
        stringRepr = line.split(delimiter)
        self.neuron_vn_id = np.where(vnIds == int(stringRepr[0]))[0][0]  #getting number of neuron that gonna be responsible for this vnId
        self.user_id = int(stringRepr[1])
        self.rating = int(stringRepr[2])/100.                     #just for lesser memory to store

'''
class VoteML:               #Removed due end of tests
    
    def __init__(self, line, delimiter):
        stringRepr = line.split(delimiter)
        self.vn_id = int(stringRepr[1])
        self.user_id = int(stringRepr[0])
        self.rating = float(stringRepr[2])/100.
'''


class UserRatings:
    '''Class to contain single user votes for VNs.
    '''
    
    def __init__(self):
        self.titles = []
        self.ratings = []
        
    def update(self, vote):
        self.titles.append(vote.neuron_vn_id)
        self.ratings.append(vote.rating)


        
class RatingsDataset(torch.utils.data.Dataset):
    '''Class to feed to VAE. Also contains other useful info.\n
    Args:
        path (str) - path to votes
        libPath (str) - path to vn
    '''
    
    def __init__(self, path = 'vndb-votes-2020-01-06', libPath = 'vn'):
        self.path = path
        self.vnIds, self.maxId = self._getUniqueVNIdsAndMax()   #get unique vn ids and max id. Last one is important for hybrid recommendation construction
        self.countOfVNs = self.vnIds.shape[0]
        self.votes = self._readVotesFromDump(self.vnIds)
        self.titles = [user.titles for user in self.votes.values()]
        self.ratings = [user.ratings for user in self.votes.values()]
        #self.vndbLib = VNDBLib(libPath)
        #self.countOfVNs = (self[0].size())[0]
        
        
    def __len__(self):
        return len(self.ratings)
    
    
    def __getitem__(self, index):
        ratingsSparse = torch.sparse.LongTensor(torch.Tensor(self.titles[index]).unsqueeze(0).to(dtype = torch.long), torch.Tensor(self.ratings[index]), torch.Size([self.countOfVNs]))
        #isRatedSparse = torch.sparse.FloatTensor(torch.Tensor(self.titles[index]).unsqueeze(0).to(dtype = torch.long), torch.Tensor(np.ones_like(self.ratings[index])), torch.Size([self.len_vnIds]))
        return ratingsSparse
    
    def _getUniqueVNIdsAndMax(self):
        '''Method to get array of all unique vn ids, which has at least one vote and max id.\n
        Returns:
            allUnique (ndarray), maxId (int)'''
        vnIds = set()
        lastId = 0
        file = open(self.path,'r')
        lines = file.readlines()
        for line in lines:
            vnId = int((line.split())[0])
            if vnId != lastId:
                vnIds.add(vnId)
                lastId = vnId
        file.close()
        return np.asarray(sorted(vnIds)), max(vnIds)
    

    def _readVotesFromDump(self, vnIds):
        '''Create dict of UserRatings.\n
        Args:
            vnIds - all unique VNids
        Returns:
            userVotes (dict of UserRatings)
        '''
        '''
        ar = np.random.randint(low = 0, high = 3, size=3)
        print(ar)
        print((np.where(ar == 1))[0], (np.where(ar == 1)))
        '''
        userVotes = dict() 
        file = open(self.path, 'r')
        lines = file.readlines()
        for line in lines:
            vote = Vote(line, ' ', vnIds)
            if True:
                if vote.user_id not in userVotes:
                    userVotes[vote.user_id] = UserRatings()
                userVotes[vote.user_id].update(vote)
        file.close()
        return userVotes
    
    def getUserLib(self, userId):
        '''Gets neuron indexes and ratings of specified user.\n
        Args:
            userId (int) - user id
        Returns:
            neuronIndexes (list), titlesAndRatings (torch.sparse.FloatTensor)
        '''
        neuronIndexes = self.votes[userId].titles
        return neuronIndexes, torch.sparse.FloatTensor(torch.Tensor(neuronIndexes).unsqueeze(0).to(dtype = torch.long), torch.Tensor(self.votes[userId].ratings), torch.Size([self.countOfVNs]))

    def compareIndexesWithNeurons(self, neuronIndexes):
        '''Gets real VN ids for neuron ids.\n
        Args:
            neuronIndexes (tensor) - neuron indexes to convert
        Returns:
            vnIds (ndarray) - requested vn ids'''
        return self.vnIds[neuronIndexes]
        

'''
class MLDataset(torch.utils.data.Dataset):
    
    def __init__(self, path = 'ml-20m/ratings.csv'):
        self.path = path
        self.vnIds = self._getUniqueVNIds()
        self.len_vnIds = self.vnIds.shape[0]
        self.votes = self._readVotesFromDump()
        self.titles = [np.searchsorted(self.vnIds, user.titles) for user in self.votes.values()]
        self.ratings = [user.ratings for user in self.votes.values()]
        print('dataset ready')
        
        
    def __len__(self):
        return len(self.ratings)
    
    
    def __getitem__(self, index):
        ratingsSparse = torch.sparse.LongTensor(torch.Tensor(self.titles[index]).unsqueeze(0).to(dtype = torch.long), torch.Tensor(self.ratings[index]), torch.Size([self.len_vnIds]))
        isRatedSparse = torch.sparse.FloatTensor(torch.Tensor(self.titles[index]).unsqueeze(0).to(dtype = torch.long), torch.Tensor(np.ones_like(self.ratings[index])), torch.Size([self.len_vnIds]))
        return isRatedSparse, ratingsSparse
    
    def _getUniqueVNIds(self):
        vnIds = set()
        lastId = 0
        file = open(self.path,'r')
        lines = file.readlines()
        for line in lines[1:]:
            vnId = int((line.split(','))[1])
            if vnId != lastId:
                vnIds.add(vnId)
                lastId = vnId
        file.close()
        return np.asarray(sorted(vnIds))
    

    def _readVotesFromDump(self):
        userVotes = dict() 
        file = open(self.path, 'r')
        lines = file.readlines()
        for line in lines[1:]:
            vote = VoteML(line,',')
            if True:
                if vote.user_id not in userVotes:
                    userVotes[vote.user_id] = UserRatings()
                userVotes[vote.user_id].update(vote)
        file.close()
        return userVotes
    
    def getUserLib(self, userId):
        userLib = self.votes[userId].titles
        neuronIndexes = userLib
        #print(neuronIndexes)
        print(self.vnIds[16318])
        return neuronIndexes, torch.sparse.FloatTensor(torch.Tensor(neuronIndexes).unsqueeze(0).to(dtype = torch.long), torch.Tensor(np.ones_like(neuronIndexes)), torch.Size([self.len_vnIds]))

    def compareIndexesWithNeurons(self, indexes):
        return self.vnIds[indexes.cpu()]
'''

def split_dataset(dataset, train_frac = 0.9):
    length = len(dataset)
    train_length = int(length * train_frac)
    test_length  = length - train_length
    return torch.utils.data.random_split(dataset, [train_length, test_length])

class SparseLinear(torch.nn.Module):
    
    def __init__(self, inputNeurons, outputNeurons):
        super(SparseLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(inputNeurons, outputNeurons))
        self.bias = torch.nn.Parameter(torch.Tensor(outputNeurons))
        self.resetParameters()
        
    def forward(self, X):
        mul = X @ self.weight
        return torch.add(mul, self.bias)
    
    def resetParameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        self.bias.data.fill_(0.01)
        
class TagTree(dict):
    '''Class that contains tag tree.\n
    Args:
        path - path to tags_parents'''
    
    def __init__(self, path = 'tags_parents'):
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                child, parent = [int(a) for a in line.split('	') if a ]
                if parent not in self:
                    self[parent] = list()
                self[parent].append(child)
                
    def selectAllChildTags(self, tagId):
        res = [tagId,]
        self._recursiveSelectOfTags(tagId, res)
        return res
    
    def _recursiveSelectOfTags(self, tagId, res):
        if tagId in self:
            for tag in self[tagId]:
                res.append(tag)
                self._recursiveSelectOfTags(tag, res)

def prepareUsersJSON(pathUsers, outputFolder, availableUserIds):
    usersDict = dict()
    with open(pathUsers, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            lineParts = line.split('	')
            currId = int(lineParts[0])
            if currId in availableUserIds:
                usersDict[lineParts[1]] = currId
    with open(os.path.join(outputFolder, 'users.json'), 'w') as file:
        file.write('var usersDict = ')
        json.dump(usersDict, file)

def prepareTagsJSON(pathTags, outputFolder, availableTagIds, tagTree):
    tagsDict = dict()
    notAdultTagsDict = dict()
    adultTags = list()
    adultParentTags = [164, 2875, 2822, 161, 2292, 170, 2485, 464, 670, 1274, 1445]
    for tag in adultParentTags:
        adultTags.extend(tagTree.selectAllChildTags(tag))
    with open(pathTags, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            lineParts = line.split('	')
            currId = int(lineParts[0])
            if not (lineParts[6] == 'f' or lineParts[7] == 'f') and currId in availableTagIds:
                tagsDict[lineParts[1]] = currId
                if currId not in adultTags and lineParts[4] != 'ero':
                    notAdultTagsDict[lineParts[1]] = currId
    with open(os.path.join(outputFolder, 'tags.json'), 'w') as file:
        file.write('var tagsDict = ')
        json.dump(tagsDict, file)
    with open(os.path.join(outputFolder, 'notAdultTags.json'), 'w') as file:
        file.write('var notAdultTagsDict = ')
        json.dump(notAdultTagsDict, file)

def prepareClientSearchJSON(pathUsers, pathTags, outputFolder, availableUserIds, availableTagIds, tagTree):
    prepareUsersJSON(pathUsers, outputFolder, availableUserIds)
    prepareTagsJSON(pathTags, outputFolder, availableTagIds, tagTree)

            

    
if(__name__ == '__main__'):
    pass