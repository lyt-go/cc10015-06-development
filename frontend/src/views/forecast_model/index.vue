<template>
  <section class="page" data-module="forecast_model">
    <header class="page-head">
      <div>
        <h2>预测模型</h2>
        <p class="page-desc">训练批次与预测结果表（V006）：每个预测点绑定训练批次与指标口径，越限按阈值拒绝，中断仅补写未完成周期。</p>
      </div>
      <div class="page-actions">
        <button class="btn primary" data-testid="open-create" type="button" @click="openCreate">登记模型</button>
        <button class="btn" data-testid="check-consistency" type="button" @click="runCheck">一致性核对</button>
        <button class="btn" data-testid="export-results" type="button" @click="exportResults">导出全份预测结果表</button>
      </div>
    </header>

    <div class="stat-row">
      <article v-for="item in stats" :key="item.label" class="stat-card">
        <span class="stat-label">{{ item.label }}</span>
        <strong class="stat-value">{{ item.value }}</strong>
      </article>
    </div>

    <h3>模型列表</h3>
    <table class="data-table">
      <thead>
        <tr>
          <th v-for="column in modelColumns" :key="column">{{ column }}</th>
          <th>状态</th>
          <th>可执行动作</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in models" :key="String(row.id)" :data-testid="`model-${row.id}`">
          <td v-for="column in modelColumns" :key="column">{{ row[column] ?? '—' }}</td>
          <td>{{ row.status }}</td>
          <td class="row-actions">
            <button class="link" type="button" :data-testid="`train-${row.id}`" @click="openTrain(row)">训练</button>
            <button class="link" type="button" :data-testid="`resume-${row.id}`" @click="resume(row)">续训</button>
            <button class="link" type="button" :data-testid="`detail-${row.id}`" @click="openDetail(row)">详情</button>
          </td>
        </tr>
        <tr v-if="!models.length">
          <td :colspan="modelColumns.length + 2" class="empty-state">暂无模型配置，可先登记模型</td>
        </tr>
      </tbody>
    </table>

    <h3>预测结果表</h3>
    <form class="filter-bar" @submit.prevent="reloadResults">
      <label class="filter-item">
        <span>模型编号</span>
        <input v-model="resultFilters.模型编号" data-testid="filter-模型编号" placeholder="按模型编号检索" />
      </label>
      <label class="filter-item">
        <span>批次编号</span>
        <input v-model="resultFilters.批次编号" data-testid="filter-批次编号" placeholder="按训练批次检索" />
      </label>
      <label class="filter-item">
        <span>消费场景</span>
        <select v-model="resultFilters.消费场景" data-testid="filter-消费场景">
          <option value="">最新批次</option>
          <option value="对比">双批对比</option>
        </select>
      </label>
      <button class="btn" data-testid="query-results" type="submit">查询</button>
    </form>
    <table class="data-table" data-testid="forecast-result-table">
      <thead>
        <tr>
          <th v-for="column in resultColumns" :key="column">{{ column }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in results" :key="String(row.id)">
          <td v-for="column in resultColumns" :key="column">{{ row[column] ?? '—' }}</td>
        </tr>
        <tr v-if="!results.length">
          <td :colspan="resultColumns.length" class="empty-state">暂无预测结果，请先在模型列表执行训练</td>
        </tr>
      </tbody>
    </table>

    <h3>留档核对（V006）</h3>
    <table class="data-table" data-testid="archive-table">
      <thead>
        <tr>
          <th v-for="column in archiveColumns" :key="column">{{ column }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in archives" :key="String(row.id)">
          <td v-for="column in archiveColumns" :key="column">{{ row[column] ?? '—' }}</td>
        </tr>
        <tr v-if="!archives.length">
          <td :colspan="archiveColumns.length" class="empty-state">暂无留档记录</td>
        </tr>
      </tbody>
    </table>

    <div v-if="showCreate" class="modal-backdrop" data-testid="create-modal">
      <form class="modal" @submit.prevent="submitCreate">
        <h3>登记模型</h3>
        <label v-for="field in modelColumns" :key="field" class="create-field">
          <span>{{ field }}</span>
          <input v-model="createValues[field]" :data-testid="`create-${field}`" />
        </label>
        <div class="page-actions">
          <button class="btn" type="button" @click="showCreate = false">取消</button>
          <button class="btn primary" data-testid="submit-create" type="submit" :disabled="creating">
            {{ creating ? '提交中…' : '确认登记' }}
          </button>
        </div>
      </form>
    </div>

    <div v-if="showTrain" class="modal-backdrop" data-testid="train-modal">
      <form class="modal" @submit.prevent="submitTrain">
        <h3>训练模型 {{ trainTarget?.模型编号 }}</h3>
        <label class="create-field">
          <span>历史数据（逗号分隔）</span>
          <input v-model="trainForm.历史数据" data-testid="train-历史数据" placeholder="如 0.10,0.12,0.11,0.13" required />
        </label>
        <label class="create-field">
          <span>业务周期</span>
          <input v-model="trainForm.业务周期" data-testid="train-业务周期" placeholder="如 2026-11" />
        </label>
        <label class="create-field">
          <span>冲突策略</span>
          <select v-model="trainForm.冲突策略" data-testid="train-冲突策略">
            <option value="replace">最新批次替换（旧批次归档）</option>
            <option value="coexist">双批并存（按消费场景选择）</option>
          </select>
        </label>
        <label class="create-field">
          <span>阈值上限</span>
          <input v-model="trainForm.阈值上限" data-testid="train-阈值上限" placeholder="越过上限按阈值拒绝" />
        </label>
        <label class="create-field">
          <span>阈值下限</span>
          <input v-model="trainForm.阈值下限" data-testid="train-阈值下限" placeholder="默认 0" />
        </label>
        <label class="create-field">
          <span>模拟中断期次</span>
          <input v-model="trainForm.模拟中断期次" data-testid="train-模拟中断期次" placeholder="留空则训练到完成" />
        </label>
        <div class="page-actions">
          <button class="btn" type="button" @click="showTrain = false">取消</button>
          <button class="btn primary" data-testid="submit-train" type="submit" :disabled="training">
            {{ training ? '训练中…' : '开始训练' }}
          </button>
        </div>
      </form>
    </div>

    <div v-if="showDetail && detail" class="modal-backdrop" data-testid="detail-modal">
      <div class="modal">
        <h3>模型详情 {{ detail.模型编号 }}</h3>
        <p class="page-desc">状态 {{ detail.status }} ｜ 当前训练批次 {{ detail.训练批次 || '—' }} ｜ 评估误差 {{ detail.评估误差 || '—' }}</p>
        <h3>训练批次</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th v-for="column in batchColumns" :key="column">{{ column }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="batch in detail.batches || []" :key="String(batch.id)">
              <td v-for="column in batchColumns" :key="column">{{ batch[column] ?? '—' }}</td>
            </tr>
            <tr v-if="!(detail.batches || []).length">
              <td :colspan="batchColumns.length" class="empty-state">暂无训练批次</td>
            </tr>
          </tbody>
        </table>
        <h3>预测结果表</h3>
        <table class="data-table" data-testid="detail-result-table">
          <thead>
            <tr>
              <th v-for="column in resultColumns" :key="column">{{ column }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in detail.results || []" :key="String(row.id)">
              <td v-for="column in resultColumns" :key="column">{{ row[column] ?? '—' }}</td>
            </tr>
            <tr v-if="!(detail.results || []).length">
              <td :colspan="resultColumns.length" class="empty-state">暂无预测结果</td>
            </tr>
          </tbody>
        </table>
        <div class="page-actions">
          <button class="btn" type="button" @click="showDetail = false">关闭</button>
        </div>
      </div>
    </div>

    <footer class="page-foot">
      <span>共 {{ models.length }} 个模型、{{ results.length }} 个预测点、{{ archives.length }} 条留档</span>
      <span v-if="notice" class="stat-label">{{ notice }}</span>
      <span v-if="errorMessage" class="error-text">{{ errorMessage }}</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

import { request } from '@/api/client'

type Row = Record<string, any>

const ENDPOINT = '/api/forecast_model'
const modelColumns = ["模型编号", "预测对象", "历史窗口", "预测周期", "模型类型", "参数配置", "训练批次", "评估误差"]
const resultColumns = ["批次编号", "模型编号", "指标编号", "口径版本", "精度位数", "业务周期", "预测期次", "预测值"]
const batchColumns = ["批次编号", "业务周期", "status", "预测周期数", "已完成期次", "拒绝点数", "评估误差", "冲突策略"]
const archiveColumns = ["留档编号", "类型", "模型编号", "批次编号", "说明", "created_at"]

const models = ref<Row[]>([])
const results = ref<Row[]>([])
const archives = ref<Row[]>([])
const resultFilters = ref<Record<string, string>>({ 模型编号: '', 批次编号: '', 消费场景: '' })
const errorMessage = ref('')
const notice = ref('')

const showCreate = ref(false)
const creating = ref(false)
const createValues = ref<Record<string, string>>({})

const showTrain = ref(false)
const training = ref(false)
const trainTarget = ref<Row | null>(null)
const trainForm = ref<Record<string, string>>({})

const showDetail = ref(false)
const detail = ref<Row | null>(null)

const stats = computed(() => [
  { label: '模型总数', value: models.value.length },
  { label: '预测点', value: results.value.length },
  { label: '留档记录', value: archives.value.length },
  { label: '已就绪模型', value: models.value.filter((m) => m.status === '已就绪').length },
])

async function readJson(response: Response) {
  const payload = await response.json()
  if (!response.ok) throw new Error(payload.detail || payload.message || `接口返回 ${response.status}`)
  return payload
}

async function reloadModels() {
  const payload = await readJson(await request(ENDPOINT))
  models.value = payload.items ?? []
}

async function reloadResults() {
  const query = new URLSearchParams(
    Object.fromEntries(Object.entries(resultFilters.value).filter(([, v]) => v)),
  ).toString()
  const payload = await readJson(await request(`${ENDPOINT}/results?${query}`))
  results.value = payload.items ?? []
}

async function reloadArchives() {
  const payload = await readJson(await request(`${ENDPOINT}/archives`))
  archives.value = payload.items ?? []
}

async function reloadAll() {
  errorMessage.value = ''
  try {
    await Promise.all([reloadModels(), reloadResults(), reloadArchives()])
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '数据读取失败'
  }
}

function openCreate() {
  createValues.value = Object.fromEntries(modelColumns.map((field) => [field, '']))
  errorMessage.value = ''
  showCreate.value = true
}

async function submitCreate() {
  creating.value = true
  errorMessage.value = ''
  try {
    const payload = await readJson(await request(ENDPOINT, {
      method: 'POST', body: JSON.stringify({ values: createValues.value }),
    }))
    if (payload.ok === false) throw new Error(payload.message || '登记失败')
    showCreate.value = false
    await reloadAll()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '登记失败'
  } finally {
    creating.value = false
  }
}

function openTrain(row: Row) {
  trainTarget.value = row
  trainForm.value = { 历史数据: '', 业务周期: '', 冲突策略: 'replace', 阈值上限: '', 阈值下限: '', 模拟中断期次: '' }
  errorMessage.value = ''
  showTrain.value = true
}

async function submitTrain() {
  if (!trainTarget.value) return
  training.value = true
  errorMessage.value = ''
  notice.value = ''
  try {
    const payload = await readJson(await request(`${ENDPOINT}/${trainTarget.value.id}/train`, {
      method: 'POST', body: JSON.stringify({ values: trainForm.value }),
    }))
    if (payload.ok === false) throw new Error(payload.message || '训练失败')
    notice.value = payload.message || '训练完成'
    showTrain.value = false
    await reloadAll()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '训练失败'
  } finally {
    training.value = false
  }
}

async function resume(row: Row) {
  errorMessage.value = ''
  notice.value = ''
  try {
    const payload = await readJson(await request(`${ENDPOINT}/${row.id}/resume`, {
      method: 'POST', body: JSON.stringify({ values: {} }),
    }))
    if (payload.ok === false) throw new Error(payload.message || '续训失败')
    notice.value = payload.message || '续训完成'
    await reloadAll()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '续训失败'
  }
}

async function openDetail(row: Row) {
  errorMessage.value = ''
  try {
    detail.value = await readJson(await request(`${ENDPOINT}/${row.id}`))
    showDetail.value = true
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '详情读取失败'
  }
}

async function runCheck() {
  errorMessage.value = ''
  notice.value = ''
  try {
    const payload = await readJson(await request(`${ENDPOINT}/consistency_check`, { method: 'POST' }))
    notice.value = payload.message || '一致性核对完成'
    await reloadAll()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '一致性核对失败'
  }
}

async function exportResults() {
  errorMessage.value = ''
  try {
    const response = await request(`${ENDPOINT}/results/export`)
    if (!response.ok) throw new Error('导出失败，请稍后重试')
    const url = URL.createObjectURL(await response.blob())
    const link = document.createElement('a')
    link.href = url
    link.download = 'forecast_result_v006.csv'
    link.click()
    URL.revokeObjectURL(url)
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '导出失败'
  }
}

onMounted(reloadAll)
</script>
