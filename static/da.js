const renderCard = async ({ target: { files } }) => {
    if (files[0]) {
        const image = await dataURL2Base64(files[0]);
        const element = document.createElement("div");
        element.classList.add("card");
        element.style.backgroundImage = `url(${image})`;
        inputbox.insertAdjacentElement("beforebegin", element);
    }
};
