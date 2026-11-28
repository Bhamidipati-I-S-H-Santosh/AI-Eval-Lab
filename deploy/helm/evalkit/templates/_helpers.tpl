{{- define "evalkit.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "evalkit.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "evalkit.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "evalkit.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "evalkit.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/* Shared pod pieces for the Deployment and the CronJobs */}}
{{- define "evalkit.envFrom" -}}
{{- if .Values.existingSecret }}
envFrom:
  - secretRef:
      name: {{ .Values.existingSecret }}
{{- end }}
env:
  - name: EVALKIT_AUDIT_LOG
    value: /data/audit/audit.jsonl
  - name: HOME
    value: /data
{{- with .Values.extraEnv }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{- define "evalkit.volumes" -}}
volumes:
  - name: config
    configMap:
      name: {{ include "evalkit.fullname" . }}
      items:
        - {key: server.yaml, path: server.yaml}
        {{- range $name, $_ := .Values.evalConfigs }}
        - {key: {{ printf "config-%s" $name | quote }}, path: {{ printf "configs/%s" $name | quote }}}
        {{- end }}
  - name: data
    {{- if .Values.persistence.enabled }}
    persistentVolumeClaim:
      claimName: {{ include "evalkit.fullname" . }}
    {{- else }}
    emptyDir: {}
    {{- end }}
  - name: tmp
    emptyDir: {}
{{- end -}}

{{- define "evalkit.mounts" -}}
volumeMounts:
  - {name: config, mountPath: /etc/evalkit, readOnly: true}
  - {name: data, mountPath: /data}
  - {name: tmp, mountPath: /tmp}
{{- end -}}
