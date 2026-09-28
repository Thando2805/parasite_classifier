async function runModelInference(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        const summaryDiv = document.getElementById('detectionSummary');
        const bannerTitle = document.getElementById('bannerTitle');
        const statusBanner = document.getElementById('statusBanner');

        if (data.success && data.detections.length > 0) {
            bannerTitle.innerText = `${data.detections[0].class.toUpperCase()} DETECTED`;
            statusBanner.className = "bg-emerald-600 text-white p-4 rounded-xl shadow-md flex items-center space-x-3";

            // Render the backend's bounding-box annotated image into the right card!
            if (data.annotated_image) {
                document.getElementById('annotatedResultImg').src = data.annotated_image;
            }

            summaryDiv.innerHTML = data.detections.map(det => `
                <div class="flex justify-between items-center bg-slate-50 p-2 rounded border border-slate-100">
                    <span class="font-bold text-emerald-800">${det.class}</span>
                    <span class="bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded text-[10px] font-semibold">Conf: ${(det.confidence * 100).toFixed(1)}%</span>
                </div>
            `).join('');
        } else {
            bannerTitle.innerText = "NO THREAT DETECTED";
            statusBanner.className = "bg-slate-700 text-white p-4 rounded-xl shadow-md flex items-center space-x-3";
            summaryDiv.innerHTML = `<p class="text-slate-500 text-center py-2">Sample appears clear of targeted anomalies.</p>`;
        }

        switchScreen('screen-results');

    } catch (error) {
        console.error("Inference Error:", error);
        alert("Failed to connect to the backend analysis server.");
        switchScreen('screen-session');
    }
}