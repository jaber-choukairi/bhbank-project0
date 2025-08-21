document.getElementById('loginForm').addEventListener('submit', async function(event) {
  event.preventDefault();

  const email = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value.trim();
  const errorMessage = document.getElementById('errorMessage');

  try {
    const response = await fetch('/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });

    if (response.ok) {
      window.location.href = '/dashboard';
    } else if (response.status === 401) {
      errorMessage.textContent = "Email ou m    ot de passe incorrect.";
      errorMessage.style.display = 'block';
    } else {
      errorMessage.textContent = "Erreur serveur, veuillez réessayer plus tard.";
      errorMessage.style.display = 'block';
    }
  } catch (error) {
    errorMessage.textContent = "Erreur réseau, veuillez vérifier votre connexion.";
    errorMessage.style.display = 'block';
  }
});
document.getElementById("start-camera").addEventListener("click", async () => {
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const faceInput = document.getElementById("face_image");

    // Démarrer la caméra
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.style.display = "block";

    // Capturer l’image après 3 secondes
    setTimeout(() => {
        const context = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Extraire image en base64
        const imageData = canvas.toDataURL("image/jpeg");
        faceInput.value = imageData;

        // Optionnel : éteindre caméra
        stream.getTracks().forEach(track => track.stop());
        video.style.display = "none";
    }, 3000);
});
