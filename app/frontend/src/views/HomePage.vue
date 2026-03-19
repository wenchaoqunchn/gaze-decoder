<template>
    <div class="search-area">
        <SearchBox></SearchBox>
    </div>
    <div class="main-box">
        <div class="poster">
            <h1 class="aoi">Library Gallery</h1>
            <PosterPlayer />
        </div>
        <div class="wrapper">
            <div class="news">
                <NewsList />
            </div>
        </div>
    </div>
    <div class="aoi footer">
        <div>National Library | Capital Library | Chinese Academy of Sciences | CALIS | BALIS Training Center | NSTL
        </div>
        <div>Copyright ©2019 &nbsp; University Library | Director's Mailbox: lib@library.edu</div>
    </div>
</template>

<script>
import NewsList from '../components/NewsList.vue';
import PosterPlayer from '../components/PosterPlayer.vue';
import SearchBox from '../components/SearchBox.vue';

export default {
    name: 'HomePage',
    components: {
        NewsList,
        PosterPlayer,
        SearchBox
    }
};
</script>

<style scoped>
h1 {
    margin: 0 0 10px 0;
    font-size: 36px;
}

.main-box {
    background-color: rgba(255, 255, 255, 0.6);
    height: fit-content;
    padding: 30px 150px;
    display: flex;
    margin: 10px 0;
}

.wrapper {
    flex: 1.5;
    width: 100%;
    position: relative;
    background-color: white;
    border-radius: 10px;
    margin: 0 60px;
}

.news {
    position: absolute;
    width: 100%;
}

.poster {
    flex: 1;
    margin: 0 20px;
}

.footer {
    margin-top: 20px;
    padding: 5px;
    background-color: rgba(0, 0, 0, 0.457);
    color: rgb(205, 205, 205);
}
</style>