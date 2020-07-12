var extendedSettings = {'likes':{'type':'likes', 'isTagSearch': true, 'search_id':"likesSearch", 'rangesRequired':true, 'startPosition':2},
                    'dislikes':{'type':'dislikes', 'isTagSearch': true,'search_id':"dislikesSearch", 'rangesRequired':false, 'startPosition':2},
                    'users':{'type':'users', 'isTagSearch': false, 'search_id':"usersSearch", 'rangesRequired':false, 'startPosition':2}};

function getTagify(input, extendedSettings, typeOfTagify, maxTagsNum, dict, normal_sort, placeholder){
    var tagify = new Tagify(input, {whitelist: dict, enforceWhitelist: true, skipInvalid: true, editTags: null, maxTags:maxTagsNum, placeholder: placeholder,
        dropdown: {enabled: 1, fuzzySearch: normal_sort}, autoComplete: {enabled: extendedSettings[typeOfTagify]['isTagSearch'], rightKey:true}}, extendedSettings[typeOfTagify])
    return tagify
}

function  usernameInputElementHide(inputSpanId){
    $('[id="'+inputSpanId+'"]').css('visibility','hidden');
    $('[id="'+inputSpanId+'"]').blur();
    $('[id="submitButton"]').removeAttr('disabled');
    $('[id="submitButton"]').addClass('buttonSubmitEnabled');
    $('[id="submitButton"]').removeClass('buttonSubmitDisabled');
}

function  usernameInputElementShow(inputSpanId){
    $('[id='+inputSpanId+']').css('visibility','visible');
    $('[id="submitButton"]').attr('disabled', 'disabled');
    $('[id="submitButton"]').removeClass('buttonSubmitEnabled');
    $('[id="submitButton"]').addClass('buttonSubmitDisabled');
}

function getKeyByValue(object, value) {
    return Object.keys(object).find(key => object[key] === value);
}

function restoreTags(tagRestoreLikesJSON, tagRestoreDislikesJSON, userId, tagifyLikes, tagifyDislikes, tagifyUsers){
    tagKeyValueLikes = JSON.parse(tagRestoreLikesJSON)
    tagKeyDislikes = JSON.parse(tagRestoreDislikesJSON)
    tagStrArr = []
    for(let i = 0; i < tagKeyValueLikes[0].length; i++)
        tagStrArr.push(getKeyByValue(tagsDict, tagKeyValueLikes[0][i]))
    tagifyLikes.loadOriginalValues(tagStrArr)
    tagStrArr = []
    for(let i = 0; i < tagKeyDislikes.length; i++)
        tagStrArr.push(getKeyByValue(tagsDict, tagKeyDislikes[i]))
    tagifyDislikes.loadOriginalValues(tagStrArr)
    for(let i = 0; i < tagKeyValueLikes[0].length; i++)
    {
        element = $("[tag_id ='"+tagKeyValueLikes[0][i]+"'][type='likes']").children(":first")
        element.attr('value', tagKeyValueLikes[1][i])
    }
    if(userId != - 1)
    {
        $("#noUser").prop("checked", false);
        username = getKeyByValue(usersDict, parseInt(userId))
        tagifyUsers.loadOriginalValues(username)
    }
    else
    {
        $("#noUser").prop("checked", true);
        tagifyUsers.removeTag()
        $("#usersSearch").attr("contenteditable","False");
        $("#usersSearch").attr("data-placeholder","No input then. But I need at least one relevant tag.");
        $("#VAEWeightInput").prop("disabled", true);
        $("#VAEWeightInput").addClass("disabledSlider");
        $("#VAEWeightInput").val(0.1);
        $('[id="submitButton"]').addClass('buttonSubmitEnabled');
        $('[id="submitButton"]').removeClass('buttonSubmitDisabled');
        $('[id="submitButton"]').removeAttr('disabled');
    }
    $("#isAdult").prop("checked", isAdult);
    $("#requestedLanguageSwitch").val(requestedLanguage);
    $('[id="submitButton"]').removeAttr('disabled');

    /*
    if(tagifyUsers.settings.maxTags == tagifyUsers.value.length)
    {
      $('#usersSearch').attr("data-placeholder",this.settings.placeholder);
    }
    */
}

function getLikesTagData(){
    tags = $("[type='likes']");
    resKeyValue = [[],[]];
    $.each(tags, function() {
        resKeyValue[0].push(parseInt($(this).attr('tag_id')))
        resKeyValue[1].push(parseFloat($(this).children(":first").val()))
    });
    return JSON.stringify(resKeyValue)
}

function getDislikesTagData(){
    tags = $("[type='dislikes']")
    resArr = []
    $.each(tags, function() {
        resArr.push(parseInt($(this).attr('tag_id')))
    });
    return JSON.stringify(resArr)
}

function getUserData(){
    user = $("[type='users']")
    return parseInt(user.attr('user_id'))
}

function checkNoUserState()
    {
        resetForm()
        if($("#noUser").is(':checked')) 
        {
            tagifyUsers.removeTag()
            $("#usersSearch").attr("contenteditable","False");
            $("#usersSearch").attr("data-placeholder","No input then. But I need at least one relevant tag.");
            $("#VAEWeightInput").prop("disabled", true);
            $("#VAEWeightInput").addClass("disabledSlider");
            $("#VAEWeightInput").val(0.1);
            disableSubmitWithNoUser();
        }
        else
        {
            $("#usersSearch").attr("contenteditable","");
            $("#usersSearch").attr("data-placeholder",tagifyUsers.settings.placeholder);
            $("#VAEWeightInput").prop("disabled", false);
            $("#VAEWeightInput").removeClass("disabledSlider");
            $("#VAEWeightInput").val(0.5);
            disableSubmitWithNoUser();
        }
    }

function enableSubmitWithNoUser()
{
    $('[id="submitButton"]').addClass('buttonSubmitEnabled');
    $('[id="submitButton"]').removeClass('buttonSubmitDisabled');
    $('[id="submitButton"]').removeAttr('disabled');
}

function disableSubmitWithNoUser()
{
    $('[id="submitButton"]').removeClass('buttonSubmitEnabled');
    $('[id="submitButton"]').addClass('buttonSubmitDisabled');
    $('[id="submitButton"]').attr('disabled', 'disabled');
}

function toggleAdultContentSwitch()
{
    resetForm()
    tagifyLikes.destroy()
    tagifyDislikes.destroy()
    $("#likesSearch").remove()
    $("#dislikesSearch").remove()
    $("#tagify__input--outside").remove()
    /*
    $("#likesInput").removeClass('tagify__input--outside')
    $("#dislikesInput").removeClass('tagify__input--outside')
    */
    
    if($("#isAdult").is(':checked') == false)
        {
            console.log('ch')
            var input = document.getElementById('likesInput')
            tagifyLikes = getTagify(input, extendedSettings, 'likes', 10, notAdultTagsDictChanged, true, 'What do you like?')
            tagifyLikes.DOM.input.classList.add('tagify__input--outside')
            tagifyLikes.DOM.scope.parentNode.insertBefore(tagifyLikes.DOM.input, tagifyLikes.DOM.scope)
            input = document.getElementById('dislikesInput')
            tagifyDislikes = getTagify(input, extendedSettings, 'dislikes', 10, notAdultTagsDictChanged, true, 'They are gonna be excluded')
            tagifyDislikes.DOM.input.classList.add('tagify__input--outside')
            tagifyDislikes.DOM.scope.parentNode.insertBefore(tagifyDislikes.DOM.input, tagifyDislikes.DOM.scope)
            isAdult = false
        }
    else
        {
            console.log('ad')
            var input = document.getElementById('likesInput')
            tagifyLikes = getTagify(input, extendedSettings, 'likes', 10, tagsDictChanged, true, 'What do you like?')
            tagifyLikes.DOM.input.classList.add('tagify__input--outside')
            tagifyLikes.DOM.scope.parentNode.insertBefore(tagifyLikes.DOM.input, tagifyLikes.DOM.scope)
            input = document.getElementById('dislikesInput')
            tagifyDislikes = getTagify(input, extendedSettings, 'dislikes', 10, tagsDictChanged, true, 'They are gonna be excluded')
            tagifyDislikes.DOM.input.classList.add('tagify__input--outside')
            tagifyDislikes.DOM.scope.parentNode.insertBefore(tagifyDislikes.DOM.input, tagifyDislikes.DOM.scope)
            isAdult = true
        }
    
    
}






validate = false
$("#noUser").prop("checked", false);
$("#isAdult").prop("checked", false);

tagsDictChanged = [];
for (var key in tagsDict)
    tagsDictChanged.push({value:key, tag_id:tagsDict[key]})

notAdultTagsDictChanged = [];
for (var key in notAdultTagsDict)
    notAdultTagsDictChanged.push({value:key, tag_id:notAdultTagsDict[key]})

var input = document.getElementById('likesInput')
if (typeof isAdult == 'undefined')
    var isAdult = false;

var tagifyLikes = NaN;
if (isAdult != true)
{
    tagifyLikes = getTagify(input, extendedSettings, 'likes', 10, notAdultTagsDictChanged, true, 'What do you like?')
}
else
{
    tagifyLikes = getTagify(input, extendedSettings, 'likes', 10, tagsDictChanged, true, 'What do you like?')
}
// add a class to Tagify's input element
tagifyLikes.DOM.input.classList.add('tagify__input--outside');

// re-place Tagify's input element outside of the  element (tagify.DOM.scope), just before it
tagifyLikes.DOM.scope.parentNode.insertBefore(tagifyLikes.DOM.input, tagifyLikes.DOM.scope);

input = document.getElementById('dislikesInput')

var tagifyDislikes = NaN;
if (isAdult != true)
{
    tagifyDislikes = getTagify(input, extendedSettings, 'dislikes', 10, notAdultTagsDictChanged, true, 'They are gonna be excluded')
}
else
{
    tagifyDislikes = getTagify(input, extendedSettings, 'dislikes', 10, tagsDictChanged, true, 'They are gonna be excluded')
}
// add a class to Tagify's input element
tagifyDislikes.DOM.input.classList.add('tagify__input--outside');

// re-place Tagify's input element outside of the  element (tagify.DOM.scope), just before it
tagifyDislikes.DOM.scope.parentNode.insertBefore(tagifyDislikes.DOM.input, tagifyDislikes.DOM.scope);


var tagInputValues = [function(){return tagifyLikes.value}, function(){return tagifyDislikes.value}]

usersDictChanged = [];
for (var key in usersDict)
    usersDictChanged.push({value:key, user_id:usersDict[key]})

input = document.getElementById('usersInput')
if(typeof userId != 'undefined' && userId != -1)
{
    tagifyUsers = getTagify(input, extendedSettings, 'users', 1, usersDictChanged, false, '')
}
else
{
    tagifyUsers = getTagify(input, extendedSettings, 'users', 1, usersDictChanged, false, 'Your VNDB username')
}
$('[id=usersSearch]').addClass('centerInput')

$("#hideMe").hide();
/*
if(tagifyUsers.value.length == 1)
{
    usernameInputElementHide('usersInput');
    $("#usersSearch").attr("contenteditable","False");
    $("#usersSearch").attr("data-placeholder","");
}

if(tagifyLikes.value.length == 10)
{
    $("#likesSearch").attr("contenteditable","False");
    $("#likesSearch").attr("data-placeholder","Maximum reached");
}

if(tagifyDislikes.value.length == 10)
{
    $("#dislikesSearch").attr("contenteditable","False");
    $("#dislikesSearch").attr("data-placeholder","Maximum reached");
}
*/

if (typeof res == 'undefined')
{
    tagifyUsers.removeTag()
    tagifyLikes.removeAllTags()
    tagifyDislikes.removeAllTags()
    $("#noUser").attr('checked', false)
    $("#isAdult").attr('checked', false)
    $("#requestedLanguageSwitch").val("-2")
    $("#VAEWeightInput").val('0.5')
    $("#spoilerLevelInput").val('0')
}


validate = true
/*
$(function () {
$('#usersInput').autocomplete({
    value: 'id',
    items: usersDictChanged
})
})
*/


//tagifyUsers.DOM.input.classList.add('tagify__userInput');

// re-place Tagify's input element outside of the  element (tagify.DOM.scope), just before it
//tagifyUsers.DOM.scope.parentNode.insertBefore(tagifyUsers.DOM.input, tagifyUsers.DOM.scope);