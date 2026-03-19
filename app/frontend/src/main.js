import { createApp } from 'vue';
import App from './App.vue';
import ElementPlus from 'element-plus';
import router from './router';
import 'element-plus/dist/index.css'; // 引入 Element Plus 样式
import './style.css';
import store from '../store';

const app = createApp(App);
app.use(ElementPlus); // 使用 Element Plus
app.use(router);
app.use(store);
app.mount('#app');