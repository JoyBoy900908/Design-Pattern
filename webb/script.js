var ws = new WebSocket("ws://127.0.0.1:8080/socket")

$(document).ready(e => {

    ws.onopen = function (event) {
        console.log("connecting");
    }

    ws.onmessage = (e) => {
        let items = JSON.parse(e.data)

        console.log(items[0])

        let timer = new Date().getTime();

        if (items[0] != "None") {
            let botmessage = '<div class="botArea"><pre class="botchat">pattern :' + items[2] + '</pre></div>'
            $(".conversationArea").html(botmessage + $(".conversationArea").html())
            botmessage = '<div class="botArea"><pre class="botchat">' + items[0] + '</pre></div>'
            $(".conversationArea").html(botmessage + $(".conversationArea").html())
            botmessage = '<div class="botArea"><pre class="botchat">' + items[1] + '</pre></div>'
            $(".conversationArea").html(botmessage + $(".conversationArea").html())
            originalImgTag = `
            <img src="..\\custom_project\\original.png"></img>
            `
            reviseImgTag = `
            <img src="..\\custom_project\\revise.png?${timer}"></img>
            `
            $(".oriuml").html(originalImgTag)
            $(".revuml").html(reviseImgTag)
        }
        else {
            let botmessage = '<div class="botArea"><pre class="botchat">' + items[1] + '</pre></div>'
            $(".conversationArea").html(botmessage + $(".conversationArea").html())

            originalImgTag = `
            <img src="..\\custom_project\\original.png"></img>
            `
            reviseImgTag = `
            <img src="..\\custom_project\\revise.png?${timer}"></img>
            `
            $(".oriuml").html(originalImgTag)
            $(".revuml").html(reviseImgTag)
        }

    }

    ws.onclose = (e) => {
        ws.close(1000, "success")
    }


    $(".submitButton").click(() => {
        console.log("press button");
        console.log($(".inputfield").val())
        ws.send(JSON.stringify($(".userInput").val()))
        let item = '<div class="userArea"><pre class="userchat">' + $(".userInput").val() + '</pre></div>'
        $(".conversationArea").html(item + $(".conversationArea").html())
        $(".userInput").val("")
        return false
    })


})

