/*
var usersKeys = Object.keys(usersDict);
var tagsKeys = Object.keys(tagsDict);
*/

$(document).ready(function() {
    $(window).keydown(function(event){
      if(event.keyCode == 13) {
        event.preventDefault();
        return false;
      }
    });
  });

function sliderChange(val, outputId) {
    document.getElementById(outputId).innerHTML = val;
}

function showCoverArt(element, url) {
    element.innerHTML = "<img src = '" + url + "'>";
}

/*
function sendData(){
    if(tagifyUsers.value.length > 0)
    {
    likesJSON = getLikesTagData()
    dislikesJSON = getDislikesTagData()
    userId = getUserData()
    $('<input>').attr({
        type: 'hidden',
        name: 'userId',
        value: userId
    }).appendTo('#recommenderForm');
    $('<input>').attr({
        type: 'hidden',
        name: 'likesJSON',
        value: likesJSON
    }).appendTo('#recommenderForm');
    $('<input>').attr({
        type: 'hidden',
        name: 'dislikesJSON',
        value: dislikesJSON
    }).appendTo('#recommenderForm');
    $('<input>').attr({
        type: 'hidden',
        name: 'noUser',
        value: userId
    }).appendTo('#recommenderForm');
    $('#recommenderForm')[0].submit();
    }
    else
        alert('Enter username')
    }
*/

    function sendData(){
        var noUserChecked = $("#noUser").is(':checked')
        var isAdult = $("#isAdult").is(':checked')
        //$("#VAEWeightInput").prop("disabled", false);
        /*
        $('#submitButton').prop('disabled', true);
        if ($('#loading').length == 0)
        {
            $('<span id="loading">  Loading...</span>').insertAfter('#resetButton')
        }
        */
        if(tagifyUsers.value.length > 0 || noUserChecked)
        {
            likesJSON = getLikesTagData()
            dislikesJSON = getDislikesTagData()
            userId = getUserData()
            userId = isNaN(userId)? -1: userId
            var names = ['userId', 'likesJSON', 'dislikesJSON', 'noUser', 'isAdult']
            var values = [userId, likesJSON, dislikesJSON, noUserChecked, isAdult]
            for(var i = 0; i < names.length; i++)
            {
                if ($('#'+names[i]+'inp').length != 0)
                {
                    $('#'+names[i]+'inp').remove();
                }
                $('<input>').attr({
                    type: 'hidden',
                    name: names[i],
                    value: values[i],
                    id: names[i]+'inp'
                }).appendTo('#recommenderForm');
            }
            
            $('#recommenderForm')[0].submit();
        }
        else
            alert('Enter username')
    }


function resetForm(){
    if($("#noUser").is(':checked'))
    {
        $('[name="VAEWeight"]').val(0.1);
        disableSubmitWithNoUser();
    }
    else
    {
        $('[name="VAEWeight"]').val(0.5);
    }
    $('[name="spoilerLevel"').val(.0)
    tagifyLikes.removeAllTags()
    tagifyDislikes.removeAllTags()
    $('#likesSearch').attr("data-placeholder",tagifyLikes.settings.placeholder);
    $("#likesSearch").attr("contenteditable","");
    $('#dislikesSearch').attr("data-placeholder",tagifyDislikes.settings.placeholder);
    $("#dislikesSearch").attr("contenteditable","");
}

/*
function onSubmitForm(){
    sendData();
   }
*/

$("#recommenderForm").one("submit", submitFormFunction);

function submitFormFunction(event) {
       event.preventDefault();
       sendData();
}

function cloneValue(){
    $("#VAEWeightInputClone").val(document.getElementById("VAEWeightInput").value)
}



//tagRestDict = {7:3.0}


/*
function showVariants(inputText, typeOfContent, outputContainerId) {
    if (inputText != ''){
        var res = [];
        var resStr = "";
        if (typeOfContent == "user") {
            value = inputText.toLowerCase()
            for (let i = 0; i < usersKeys.length; i++) {
                if (usersKeys[i].toLowerCase().search(value) == 0)
                    res.push(usersKeys[i])
                if (res.length > 5) {
                    break
                }
            }
        }
        else {
            value = inputText.toLowerCase()
            for (let i = 0; i < tagsKeys.length; i++) {
                if (tagsKeys[i].toLowerCase().search(value) == 0)
                    res.push(tagsKeys[i])
                if (res.length > 5) {
                    break
                }
            }
        }
        outputElement = document.getElementById(outputContainerId)
        outputElement.innerHTML = ""
        for (let i = 0; i < res.length; i++) {
            resStr += "<span>" + res[i] + " " + usersDict[res[i]] + "</span><br>"
        }
        outputElement.innerHTML = resStr
    }
    else {
        outputElement = document.getElementById(outputContainerId)
        outputElement.innerHTML = ""
    }
}
*/