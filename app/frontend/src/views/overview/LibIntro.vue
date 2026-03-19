<template>
    <div class="main-content">
        <div class="aoi breadcrumb">
            You are here: Home > About the Library > Library Profile
        </div>
        <div class="aoi title">
            <h2>Library Profile</h2>
        </div>
        <div class="aoi content">
            <img src="../../assets/libintro.jpg" alt="Library building" class="float-image" />
            <p>Founded in 1960, the university library has evolved across several locations and now serves readers
                through two active sites: the main campus and the Tongzhou campus. The main library building opened in
                September 2019, offering more than 40,000 square meters of space and over 3,000 seats. The Tongzhou
                branch provides an additional study environment with more than 500 seats. The library follows an
                open-stack service model that integrates collection access, borrowing, and on-site reading.
            </p>

            <p>The collection includes more than two million printed volumes, over 1,600 journals and magazines, and a
                wide range of electronic resources. The holdings support engineering as a major focus while also
                covering science, social science, arts, and interdisciplinary subjects through print, digital, and
                multimedia formats.
            </p>

            <p>In addition to traditional circulation services, the library provides novelty search support, document
                delivery, interlibrary loan, citation and indexing checks, information literacy instruction, subject
                intelligence analysis, intellectual property information services, network access, and self-service
                borrowing and returns.</p>

            <p>The library also collaborates with major library and information institutions and participates in shared
                service networks such as OCLC, CALIS, and BALIS. These partnerships support cooperative cataloging,
                database acquisition, course-resource development, reference collaboration, and interlibrary borrowing
                services.
            </p>
        </div>
    </div>
</template>

<script setup>
</script>

<style scoped>
.title {
    text-align: center;
}

.main-box {
    background-color: rgba(255, 255, 255, 0.928);
    height: 95%;
    padding: 20px 200px;
    text-align: left;
    display: flex;
}

.sidebar {
    flex: 1;
}

.bar-title {
    background-size: cover;
    background-position: center;
    background-image: url(../assets/sidebar.png);
    text-align: center;
}

.main-content {
    flex: 3;
    padding: 0 5%;
}

.content {
    text-indent: 2em;
    font-size: 18px;
    line-height: 1.8;
}

.float-image {
    float: right;
    /* 使图片浮动到右侧 */
    margin-left: 20px;
    /* 图片与文本之间的间距 */
    margin-bottom: 10px;
    /* 图片下方与文本之间的间距 */
    max-width: 350px;
    /* 设置图片的最大宽度 */
    height: auto;
    /* 自适应高度 */
}
</style>