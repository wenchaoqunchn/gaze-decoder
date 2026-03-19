<template>
  <!-- <AOISelector> -->
  <div id="container">
    <div id="bg">
      <div id="tt" class="mg">Session config</div>
      <div class="mg">
        Initial session index：
        <el-input-number v-model="num" :min="0" :max="99" />
      </div>
      <div class="mg">
        <el-switch v-model="skipReq" @change="updateSkipRequest" active-text="Disable POST" inactive-text="Enable POST">
        </el-switch>
      </div>
      <div class="mg">
        <el-switch v-model="exportAOI" @change="updateExportAOI" active-text="Export AOI" inactive-text="No export">
        </el-switch>
      </div>
      <div class="mg">
        <el-button type="primary" id="btn" @click="handleClick">
          Calibrate >
        </el-button>
      </div>
    </div>
  </div>
  <!-- </AOISelector> -->
</template>

<script setup>
import axios from "axios";
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useStore } from 'vuex';
import { ElMessage } from 'element-plus';

const store = useStore();
const router = useRouter();

const num = ref(0);
const skipReq = computed({
  get: () => store.state.skipRequest,
  set: (value) => store.commit('toggleSkipRequest', value),
});

// 新增的计算属性
const exportAOI = computed({
  get: () => store.state.exportAOI,
  set: (value) => store.commit('toggleExportAOI', value),
});


// 定义处理点击事件的函数
const handleClick = () => {
  // 将 num 的值传入接口
  if (!skipReq.value) {
    axios
      .post("/api/init", { focus_session: num.value, exportAOI: exportAOI.value }) // 传递 num 和 exportAOI
      .then((response) => {
        console.log(response);
        return axios.post("/api/start");
      })
      .then((response) => {
        console.log(response);
      })
      .catch((error) => {
        console.error('Error during the API call:', error);
      });
  } else {
    ElMessage.success('API request skipped');
  }

  // 跳转到新路由
  router.push({ name: 'Calibration1Ready' });
};
</script>

<style scoped>
#tt {
  color: #444;
  font-size: 50px;
  font-weight: bold;
}

#container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  width: 100vw;
  background-color: #363636;
}

#bg {
  background-color: #ffffffdb;
  border: 1px solid rgb(192, 192, 192);
  border-radius: 20px;
  width: 600px;
  padding: 20px;
}

.mg {
  margin: 20px;
}

#btn {
  border-radius: 15px;
  font-size: 50px;
  margin-top: 20px;
  padding: 50px;
}
</style>