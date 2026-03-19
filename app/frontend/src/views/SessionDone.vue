<template>
  <div id="container">
    <div id="bg">
      <div id="tt" class="mg">Session Complete</div>
      <div class="mg">
        <el-button type="primary" class="btn" @click="handleStop">
          Stop
        </el-button>
        <el-button type="default" class="btn" @click="handleAnalyze">
          Analyze
        </el-button>
        <el-button type="default" class="btn" @click="openHomePage">
          Open Home
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ElMessage } from 'element-plus';
import axios from 'axios';
import { useStore } from 'vuex';
import { computed } from 'vue';

const store = useStore();

// use Vuex state
const skipReq = computed(() => store.state.skipRequest);

const handleStop = () => {
  if (!skipReq.value) {
    axios
      .post("/api/stop")
      .then((response) => {
        if (response.status === 200) {
          ElMessage.success('Session stopped successfully!');
        }
      })
      .catch((error) => {
        console.error('Error during the stop API call:', error);
        ElMessage.error('Stop operation failed, please try again.');
      });
  } else {
    ElMessage.success('API request skipped');
  }
};

const handleAnalyze = () => {
  if (skipReq.value === false) {
    axios
      .post("/api/analyze")
      .then((response) => {
        if (response.status === 200) {
          ElMessage.success('Analysis completed successfully!');
        }
      })
      .catch((error) => {
        console.error('Error during the analyze API call:', error);
        ElMessage.error('Analysis operation failed, please try again.');
      });
  } else {
    ElMessage.success('API request skipped');
  }
};

// open the home page in a new tab
const openHomePage = () => {
  window.open('/', '_blank');
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
  background-color: #ffffffda;
  border: 1px solid rgb(192, 192, 192);
  border-radius: 20px;
  width: 600px;
  height: 400px;
  padding: 20px;
}

.mg {
  margin: 20px;
}

.btn {
  border-radius: 15px;
  font-size: 50px;
  margin-top: 20px;
  padding: 50px;
}
</style>