<template>
    <div class="card-container">
        <el-card class="card" :class="{ 'active': activeCard === 2 }" @click="setActiveCard(2)">
            <div class="section-container">
                <div class="img-box">
                    <div class="aoi img f2">
                        <div class="layer-text">F2 / Level 2 - Social Sciences</div>
                    </div>
                </div>
                <div class="aoi section" v-for="(categories, sectionName) in infoF2" :key="sectionName">
                    <h3>{{ sectionName }}</h3>
                    <ul>
                        <li v-for="(description, code) in categories" :key="code">
                            {{ code }}: {{ description }}
                        </li>
                    </ul>
                </div>
            </div>
        </el-card>

        <el-card class="card" :class="{ 'active': activeCard === 3 }" @click="setActiveCard(3)">
            <div class="section-container">
                <div class="img-box">
                    <div class="aoi key-aoi img f3">
                        <div class="layer-text">F3 / Level 3 - Science & Tech</div>
                    </div>
                </div>
                <div class="aoi key-aoi section" v-for="(categories, sectionName) in infoF3" :key="sectionName">
                    <h3>{{ sectionName }}</h3>
                    <ul>
                        <li style="font-family: 'MingLiU'" v-for="(description, code) in categories" :key="code">
                            {{ code }}: {{ description }}
                        </li>
                    </ul>
                </div>
            </div>
        </el-card>

        <el-card class="card" :class="{ 'active': activeCard === 4 }" @click="setActiveCard(4)">
            <div class="section-container">
                <div class="img-box">
                    <div class="aoi img f4">
                        <div class="layer-text">F4 / Level 4 - Literature</div>
                    </div>
                </div>
                <div class="aoi section" v-for="(categories, sectionName) in infoF4" :key="sectionName">
                    <h3>{{ sectionName }}</h3>
                    <ul>
                        <li style="font-family: 'MingLiU'" v-for="(description, code) in categories" :key="code">
                            {{ code }}: {{ description }}
                        </li>
                    </ul>
                </div>
            </div>
        </el-card>
    </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import { useStore } from 'vuex';

const store = useStore();

// use Vuex state
const infoF2 = computed(() => store.state.staticF2);
const infoF3 = computed(() => store.state.staticF3);
const infoF4 = computed(() => store.state.staticF4);

const activeCard = ref(store.state.selectedFloor);

function setActiveCard(card) {
    activeCard.value = card;
    store.commit('setSelectedFloor', card);
    console.log(store.state.selectedFloor);
}
</script>

<style scoped>
h1 {
    margin: 0;
}

.card-container {
    display: flex;
    justify-content: space-between;
    flex-direction: column;
    padding: 0px;
    height: 60vh;
}

.card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 0px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s;
    margin-bottom: 5px;
    width: 100%;
}

:deep(.card .el-card__body) {
    padding: 0;
}

.card:last-child {
    margin-bottom: 0;
}

.card {
    transition: transform 0.2s, background-color 0.2s;
}

.card.active {
    /* transform: scale(1.05); */
    background-color: #e6f1ff;
    /* 被点击时的背景颜色 */
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}

.card.active:hover {
    /* transform: scale(1.05); */
    background-color: #e6f1ff;
}

.card:hover {
    transform: scale(1.02);
    background-color: #ececec;
}

h2 {
    margin: 0;
}

.title-box {
    text-align: center;
}


.f2 {
    background-image: url('../../assets/f3-img.jpg');
}

.f3 {
    background-image: url('../../assets/f4-img.jpg');
}

.f4 {
    background-image: url('../../assets/f2-img.jpg');
}

.img-box {
    width: 400px;
    margin-right: 60px;
}

.layer-text {
    background-color: rgba(0, 0, 0, 0.442);
    height: 50px;
    font-size: 30px;
    /* 大号字体 */
    color: rgba(230, 230, 230, 0.957);
    /* 白色字体 */
    font-weight: bold;
    /* 加粗字体 */
    padding-left: 20px;
}


.img-box .img {
    width: 100%;
    height: 100%;
    /* 设置固定高度 */
    background-size: cover;
    /* 确保背景图覆盖 */
    background-position: center;
    /* 居中显示背景图 */
}

.section-container {
    display: flex;
    flex-wrap: wrap;
}

.section {
    flex: 1;
    margin: 18px 0;
}

.section:last-child {
    margin-right: 20px;
}

h3 {
    margin: 0;
}

ul {
    padding-left: 0;
    list-style-type: none;
    margin: 0;
}

li {
    color: #686868;
}
</style>