from flask import Flask, request, render_template
import torch
import VAERecommender_hybrid as rs
import VAERecommender_update as upd
import datetime
import json
from apscheduler.schedulers.background import BackgroundScheduler
import time

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('CUDA is available,',torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
    else:
        print('DOOM')
        print(torch.version.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



randomNumber = str(int(torch.randint(10000,50000,(1,))[0])) + datetime.datetime.today().strftime("%Y%m%d")

data = rs.RecommenderData(pathVN = './dump/db/vn', pathTags_vn = './dump/db/tags_vn',\
                                 pathTags_parents = './dump/db/tags_parents', pathTags= './dump/db/tags',
                                 pathVotes = './dump/db/votes', pathUsers='./dump/db/users',
                                 pathReleases= './dump/db/releases', pathReleasesLang='./dump/db/releases_lang',\
                                pathReleasesVN= './dump/db/releases_vn', pathLanguages='./dump/languages') 
rec = rs.HybridRecommender(data)
print('model ready')
app = Flask(__name__)
print('web app ready')


def getSelectLanguageHTML():
    res = '<select name="requestedLanguage" id="requestedLanguageSwitch"><option value = "-2">Any</option>'
    for key, value in data.frontendData.languageTextRepr.items():
        if key != -1 and key != -2:
            res += '<option value = "'+str(key)+'">'+str(value)+'</option>'
    res += '<option value = "-1">Other</option></select>'
    return res

selectLanguageHTML = getSelectLanguageHTML()


@app.route('/', methods = ['POST', 'GET'])
def main():
    return render_template('recommender.html', randomNumber = randomNumber, selectLanguageHTML = selectLanguageHTML)

@app.route('/recommender', methods = ['POST'])
def recommend():
    if request.method == 'POST':
        '''
        VAEWeight = float(request.form.get('VAEWeight'))
        userId = int(request.form.get('userId'))
        dislikedTagIds = [int(x) for x in (request.form.get('dislikes')).split()]
        likedTagIds = [int(x) for x in (request.form.get('likes')).split()]
        likedTagVals = [float(x) for x in (request.form.get('likesVal')).split()]
        likesDict = dict(zip(likedTagIds, likedTagVals))
        spoilerMax = float(request.form.get('spoilerLevel'))
        '''
        VAEWeight = float(request.form.get('VAEWeight'))
        userId = int(request.form.get('userId'))
        dislikedTagIds = json.loads(request.form.get('dislikesJSON'))
        likesKeys, likesVals = json.loads(request.form.get('likesJSON'))
        likesDict = dict(zip(likesKeys, likesVals))
        spoilerMax = float(request.form.get('spoilerLevel'))
        isUnregistredUser = True if request.form.get('noUser') == 'true' else False
        isAdult = True if request.form.get('isAdult') == 'true' else False
        requestedLanguage = int(request.form.get('requestedLanguage'))
        print('NEW REQUEST', isUnregistredUser)
        if len(likesDict) <= 10 and len(dislikedTagIds) <= 10 and (isUnregistredUser or (userId in data.dataset.votes.keys())):
            userId = userId if isUnregistredUser != True else -1
            res, score = rec.recommendKItems(VAEWeight, userId, dislikedTagIds, likesDict, spoilerMax, isAdult, requestedLanguage, data, fullInfo = True)
        return render_template('recommender.html', res = res, randomNumber = randomNumber, selectLanguageHTML = selectLanguageHTML)

@app.route('/faq', methods = ['POST', 'GET'])
def faq():
    return render_template('faq.html')


if __name__ == '__main__':
    
    '''
    scheduler = BackgroundScheduler()
    scheduler.add_job(func = upd.updateModel, args = (data, rec), trigger = "cron", hour = 15, minute = 28)
    scheduler.start()
    
    app.run(debug = True, use_reloader = False)
    '''
    app.run(debug = True)