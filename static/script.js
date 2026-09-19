const uploadBtn = document.getElementById("uploadBtn");
const runBtn = document.getElementById("runBtn");

const input = document.getElementById("videoInput");

const status = document.getElementById("uploadStatus");
const progress = document.getElementById("progressFill");

const outputVideo = document.getElementById("outputVideo");
const rallyList = document.getElementById("rallyList");

const bounceCount = document.getElementById("bounceCount");
const rallyCount = document.getElementById("rallyCount");

let uploadedFile = null;


// Upload
uploadBtn.addEventListener("click", async () => {

    if (input.files.length === 0) {
        status.innerText = "Please choose a video.";
        return;
    }

    const form = new FormData();

    form.append("video", input.files[0]);

    status.innerText = "Uploading...";

    const response = await fetch("/upload", {
        method: "POST",
        body: form
    });

    const data = await response.json();

    if (data.success) {

        uploadedFile = data.filename;

        status.innerText = "Uploaded: " + data.filename;

    } else {

        status.innerText = "Upload failed.";

    }

});


// Run Detection
runBtn.addEventListener("click", async () => {

    if (uploadedFile == null) {
        alert("Upload video first.");
        return;
    }

    progress.style.width = "10%";

    status.innerText = "Running AI pipeline...";

    const response = await fetch("/run", {

        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify({
            filename: uploadedFile
        })

    });

    progress.style.width = "100%";

    const data = await response.json();

    if (!data.success) {

        status.innerText = "Pipeline failed.";

        return;

    }

    status.innerText = "Finished!";

    bounceCount.innerText = data.bounce;
    rallyCount.innerText = data.rallies.length;

    outputVideo.src = "/static/outputs/" + data.video;

    rallyList.innerHTML = "";

    data.rallies.forEach(file => {

        const div = document.createElement("div");

        div.className = "rally-item";

        div.innerHTML = `
            <a href="/static/rallies/${file}" target="_blank">
                🎥 ${file}
            </a>
        `;

        rallyList.appendChild(div);

    });

});