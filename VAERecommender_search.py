# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 23:59:42 2020

@author: caster
"""

import VAERecommender_prepare_data as ds
import math
import numpy as np

class TagSearcher:
    
    def __init__(self, vndbTagDict, tagTree, releasesData):
        self.tagDict = vndbTagDict
        self.tagTree = tagTree
        self.releasesData = releasesData
        self.maxId = max(vndbTagDict.keys())            #max id of VN that tagged. Useful when constructing hybrid recommendation
        
    def search(self, res, weight, tagsToRemove, tagsToGet, method = 1, maxSpoilerLevel = 0.):
        relevantVNIds = self._removeVNs(tagsToRemove, tagsToGet, maxSpoilerLevel)
        relevantVNIds = self._getScores(res, weight, tagsToGet, relevantVNIds, method)
        return relevantVNIds
        
    def _removeVNs(self, restrictedTags, tagsToGet, maxSpoilerLevel):
        removedVNIds = []
        if len(restrictedTags) != 0:
            extendedRestrictedTags = []
            for rtag in restrictedTags:
                extendedRestrictedTags.extend(self.tagTree.selectAllChildTags(rtag))
            for gtag in tagsToGet.keys():
                try:
                    extendedRestrictedTags.remove(gtag)
                except ValueError:
                    pass
            for vnId in self.tagDict.keys():
                if (len(tagsToGet) != 0 and all(1 if tag not in tagsToGet.keys() else 0 for tag in self.tagDict[vnId])) or\
                    any(1 for tag in self.tagDict[vnId] if (tag in extendedRestrictedTags) or (tag in tagsToGet.keys() and self.tagDict[vnId][tag]['s'] > maxSpoilerLevel)):
                    removedVNIds.append(vnId)
            
        relevantVNIds = list(set(self.tagDict).difference(set(removedVNIds)))
        return relevantVNIds
                
    def _getScoreBasedOnTagsWithoutWeights(self, vnId, requestedTags):
        res = 0
        placeholderDict = dict()
        placeholderDict['r'] = 0
        for tag in requestedTags:
            res += self.tagDict[vnId].get(tag, placeholderDict)['r']
        return res / (len(requestedTags) * 3.)
    
    def _getDistanceScore(self, vnId, requestedTags, fittingTags, max_distance, placeholderDict):
        res = 0
        for tag in requestedTags:
            res += abs(requestedTags[tag] - self.tagDict[vnId].get(tag, placeholderDict)['r'])
        return math.exp(res)
        
                
    def _getScores(self, res_fin, weight, requestedTags, relevantIds, method):
        if len(requestedTags) != 0:
            vnWithAnyRequestedTagIds = []
            max_tag_distance = 3
            max_distance = max_tag_distance * len(requestedTags)
            max_exp = math.exp(max_distance)
            placeholderDict = dict()
            placeholderDict['r'] = 0
            for vnId in relevantIds:
                fittingTags = [reqTag for reqTag in requestedTags if reqTag in self.tagDict[vnId]]
                if fittingTags:
                    '''
                    if method == 0:
                        curr_score = self._getScoreBasedOnTagsWithoutWeights(vnId, requestedTags)
                    else:
                    '''
                    res_fin[vnId] = weight * (max_exp - self._getDistanceScore(vnId, requestedTags, fittingTags, max_distance, placeholderDict))/max_exp
                    vnWithAnyRequestedTagIds.append(vnId)
            return vnWithAnyRequestedTagIds
        else:
            return relevantIds

    def removeUnrelevantTagsByTagsAndAgeRatingAndLanguages(self, fin_res, relevantVNIds, allVNIds, isAdult, requestedLanguage):
        if relevantVNIds:
            nonrelevantIds = np.array(list(set(allVNIds).difference(set(relevantVNIds))))
            fin_res[nonrelevantIds] = -1
        if not isAdult:
            fin_res[self.releasesData.nsfwVNs] = -1
        if requestedLanguage != -2:
            nonrelevantIds = np.array(list(set(allVNIds).difference(set(self.releasesData.dictOfVNsByLanguage[requestedLanguage]))))
            fin_res[nonrelevantIds] = -1
        
                        
                        
                        
if __name__ == '__main__':
    tagTree = ds.TagTree()
    vndbTagDict = ds.VNTagDict()
    se = TagSearcher(vndbTagDict, tagTree)
    print('loaded')
    #print(se.search([0], [32, 433], 1, [1.,2.]))