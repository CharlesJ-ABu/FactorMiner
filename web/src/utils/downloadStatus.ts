export type DownloadStatus = 'idle' | 'running' | 'completed' | 'failed' | 'partial_failed';

export type DownloadState = {
  status: DownloadStatus;
  progress: number;
  message: string;
};

export type DownloadProgressEvent = {
  type: string;
  task_id: string;
  status?: string;
  progress: number;
  message: string;
};

export const emptyDownloadState: DownloadState = { status: 'idle', progress: 0, message: '' };

export function isTerminalDownloadStatus(status: string): boolean {
  return status === 'completed' || status === 'failed' || status === 'partial_failed';
}

export function matchesDownloadTask(event: DownloadProgressEvent | null, taskId: string | null): boolean {
  return !!taskId && event?.type === 'download_progress' && event.task_id === taskId;
}

export function createDownloadEventTracker() {
  let pending = false;
  let taskId: string | null = null;
  let terminalReceived = false;
  const pendingEvents = new Map<string, DownloadProgressEvent>();

  function reset() {
    pending = false;
    taskId = null;
    terminalReceived = false;
    pendingEvents.clear();
  }

  function receive(event: DownloadProgressEvent | null): DownloadProgressEvent | null {
    if (event?.type !== 'download_progress' || !event.task_id) return null;
    if (pending) {
      const previous = pendingEvents.get(event.task_id);
      // Keep a terminal result even if delayed worker progress follows it.
      if (!isTerminalDownloadStatus(previous?.status ?? '')) pendingEvents.set(event.task_id, event);
      return null;
    }
    if (!matchesDownloadTask(event, taskId) || terminalReceived) return null;
    terminalReceived = isTerminalDownloadStatus(event.status ?? '');
    return event;
  }

  return {
    start() { reset(); pending = true; },
    receive,
    accept(acceptedTaskId: string): DownloadProgressEvent | null {
      taskId = acceptedTaskId;
      pending = false;
      const event = pendingEvents.get(taskId);
      pendingEvents.clear();
      return event ? receive(event) : null;
    },
    reset,
  };
}

export function reduceDownloadProgress(
  state: DownloadState, event: DownloadProgressEvent, taskId: string | null,
): DownloadState {
  if (!matchesDownloadTask(event, taskId) || isTerminalDownloadStatus(state.status)) return state;
  // Progress is a completion ratio, never evidence that the download succeeded.
  const status = event.status === 'running' || isTerminalDownloadStatus(event.status ?? '')
    ? event.status as DownloadStatus : 'running';
  return { status, progress: Math.max(0, Math.min(100, event.progress)), message: event.message };
}

export function downloadOutcome(status: DownloadStatus): { tone: 'success' | 'error' | 'warning'; message: string } | null {
  if (status === 'completed') return { tone: 'success', message: 'Successfully completed batch download!' };
  if (status === 'failed') return { tone: 'error', message: 'Batch download failed. Review the execution log before using the data.' };
  if (status === 'partial_failed') return { tone: 'warning', message: 'Batch download partially failed. Some targets were not updated.' };
  return null;
}
