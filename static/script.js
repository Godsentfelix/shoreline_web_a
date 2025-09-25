function uploadFile() {
    let fileInput = document.getElementById("fileInput");
    let file = fileInput.files[0];

    if (!file) {
        alert("Please select a file.");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    fetch("/upload", { method: "POST", body: formData })
        .then(response => response.json())
        .then(data => {
            let resultsDiv = document.getElementById("results");
            if (data.error) {
                resultsDiv.innerHTML = `<p>Error: ${data.error}</p>`;
            } else {
                resultsDiv.innerHTML = `<p>Processed: ${data.filename} - <a href="${data.shapefile}" download>Download Shapefile</a></p>`;
            }
        })
        .catch(error => console.error("Error:", error));
}
