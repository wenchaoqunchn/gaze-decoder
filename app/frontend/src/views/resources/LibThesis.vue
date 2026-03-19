<template>
    <div class="main-content">
        <div class="aoi breadcrumb">
            You are here: Home > Resources > Theses & Dissertations
        </div>
        <div class="aoi title">
            <h2>University Theses & Dissertations</h2>
        </div>
        <div class="aoi content">
            <p>The university thesis repository includes publicly available doctoral dissertations since 1997, master's
                theses since 1996, and selected outstanding undergraduate theses since 2008. Full text is available to
                on-campus users.</p>

            <p>Repository URL: <a href="https://libthesis.bjut.edu.cn" target="_blank">https://libthesis.bjut.edu.cn</a>
            </p>
            <img src="../../assets/thesis.jpg" alt="University thesis repository" class="thesis-image" />
        </div>
    </div>
</template>

<script setup>
</script>

<style scoped>
.title {
    text-align: center;
}

.main-content {
    padding: 20px 5%;
}

.content {
    text-indent: 2em;
    font-size: 18px;
    line-height: 1.6;
}

.thesis-image {
    display: block;
    /* Keep the image on its own line */
    margin: 20px auto;
    /* Center the image with vertical spacing */
    max-width: 65%;
    /* Responsive image sizing */
    height: auto;
    /* Preserve aspect ratio */
}
</style>